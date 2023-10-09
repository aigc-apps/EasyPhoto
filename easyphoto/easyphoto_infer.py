import copy
import glob
import logging
import os

import cv2
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import AutoencoderKL, ControlNetModel
from easyphoto.easyphoto_config import (DEFAULT_NEGATIVE, DEFAULT_NEGATIVE_XL,
                                        DEFAULT_POSITIVE, DEFAULT_POSITIVE_XL,
                                        SDXL_MODEL_NAME, abs_models_path,
                                        easyphoto_outpath_samples, models_path,
                                        user_id_outpath_samples,
                                        validation_prompt)
from easyphoto.easyphoto_utils import (check_files_exists_and_download,
                                       check_id_valid, save_image)
from easyphoto.face_process_utils import (Face_Skin, call_face_crop,
                                          color_transfer, crop_and_paste)
from easyphoto.sd_diffusers import (i2i_inpaint_call, t2i_sdxl_call,
                                    weight_dtype)
from easyphoto.train_kohya.utils.gpu_info import gpu_monitor_decorator
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image

def resize_image(input_image, resolution, nearest = False, crop264 = True):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    if crop264:
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
    else:
        H = int(H)
        W = int(W)
    if not nearest:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    else:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)
    return img

model_canny = None
model_openpose = None
model_tile = None
model_color = None

model_openpose_preprocess = None

def get_controlnet_preprocess(unit, input_image):
    global model_openpose_preprocess
    if unit == "canny":
        canny_image = cv2.Canny(np.array(input_image, np.uint8), 100, 200)[:, :, None]
        output_image = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))

    elif unit == "openpose":
        if model_openpose_preprocess is None:
            model_openpose_preprocess = OpenposeDetector.from_pretrained(os.path.join(abs_models_path, "Others", "bubbliiiing/controlnet_helper/", "controlnet_detector"))
        output_image = model_openpose_preprocess(input_image)
        
    elif unit == "color":
        blur_ratio      = 24
        h, w, c         = np.shape(input_image)
        color_image     = np.array(input_image, np.uint8)

        color_image     = resize_image(color_image, 1024)
        now_h, now_w    = color_image.shape[:2]

        color_image = cv2.resize(color_image, (int(now_w//blur_ratio), int(now_h//blur_ratio)), interpolation=cv2.INTER_CUBIC)  
        color_image = cv2.resize(color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST)
        color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_CUBIC)
        output_image = Image.fromarray(np.uint8(color_image))

    elif unit == "tile":
        output_image = input_image

    return output_image

def get_controlnet_unit(unit):
    global model_canny, model_openpose, model_tile, model_color
    if unit == "canny":
        if model_canny is None:
            model_canny = ControlNetModel.from_pretrained(os.path.join(abs_models_path, "Others", "bubbliiiing/controlnet_helper/controlnet", "sd-controlnet-canny"), torch_dtype=weight_dtype)
        return model_canny
    
    elif unit == "openpose":
        if model_openpose is None:
            model_openpose = ControlNetModel.from_pretrained(os.path.join(abs_models_path, "Others", "bubbliiiing/controlnet_helper/controlnet", "sd-controlnet-openpose"), torch_dtype=weight_dtype)
        return model_openpose
    
    elif unit == "color":
        if model_color is None:
            model_color = ControlNetModel.from_pretrained(os.path.join(abs_models_path, "Others", "bubbliiiing/controlnet_helper/controlnet", "sd-controlnet-color"), torch_dtype=weight_dtype)
        return model_color
    
    elif unit == "tile":
        if model_tile is None:
            model_tile = ControlNetModel.from_pretrained(os.path.join(abs_models_path, "Others", "bubbliiiing/controlnet_helper/controlnet", "sd-controlnet-tile"), torch_dtype=weight_dtype)
        return model_tile
    
    return None


def sdxl_txt2img(
    input_prompt = '1girl',
    diffusion_steps = 50,
    width: int = 1024,
    height: int = 1024,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
):
    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = t2i_sdxl_call(
        steps=diffusion_steps,
        seed=seed,

        cfg_scale=7,
        width=int(width),
        height=int(height),

        prompt=positive,
        negative_prompt=negative,
        sd_model_checkpoint=sd_model_checkpoint
    )

    return image

def inpaint(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    controlnet_pairs: list,
    input_prompt = '1girl',
    diffusion_steps = 50,
    denoising_strength = 0.45,
    hr_scale: float = 1.0,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_lora_checkpoint = [],
    sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
):
    assert input_image is not None, f'input_image must not be none'
    controlnet_units_list           = []
    controlnet_image                = []
    controlnet_conditioning_scale   = []
    w = int(input_image.width)
    h = int(input_image.height)


    for index, pair in enumerate(controlnet_pairs):
        controlnet_units_list.append(
            get_controlnet_unit(pair[0])
        )
        controlnet_image.append(
            get_controlnet_preprocess(pair[0], pair[1])
        )
        controlnet_conditioning_scale.append(
            pair[2]
        )

    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = i2i_inpaint_call(
        images=[input_image],
        mask_image=select_mask_input,
        denoising_strength=denoising_strength,

        controlnet_units_list=controlnet_units_list, 
        controlnet_image=controlnet_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,

        steps=diffusion_steps,
        seed=seed,

        cfg_scale=7,
        width=int(w*hr_scale),
        height=int(h*hr_scale),

        prompt=positive,
        negative_prompt=negative,
        sd_lora_checkpoint=sd_lora_checkpoint,
        sd_model_checkpoint=sd_model_checkpoint,
        sd_base15_checkpoint=os.path.join(models_path, "Others", "stable-diffusion-v1-5")
    )

    return image

retinaface_detection = None
image_face_fusion = None
skin_retouching = None
portrait_enhancement = None
face_skin = None
face_recognition = None
check_hash = True

def easyphoto_infer_forward(
    sd_model_checkpoint, selected_template_images, init_image, uploaded_template_images, additional_prompt, \
    before_face_fusion_ratio, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength, second_diffusion_steps, second_denoising_strength, \
    seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle, color_shift_last, super_resolution, display_score, \
    background_restore, background_restore_denoising_strength, sd_xl_input_prompt, sd_xl_resolution, tabs, *user_ids,
):
    # global
    global retinaface_detection, image_face_fusion, skin_retouching, portrait_enhancement, face_skin, face_recognition, check_hash

    # check & download weights of basemodel/controlnet+annotator/VAE/face_skin/buffalo/validation_template
    check_files_exists_and_download(check_hash)
    check_hash = False

    for user_id in user_ids:
        if user_id != "none":
            if not check_id_valid(user_id, user_id_outpath_samples, models_path):
                return "User id is not exist", [], []  
    
    # update donot delete but use "none" as placeholder and will pass this face inpaint later
    passed_userid_list = []
    for idx, user_id in enumerate(user_ids):
        if user_id == "none":
            passed_userid_list.append(idx)

    if len(user_ids) == len(passed_userid_list):
        return "Please choose a user id.", [], []

    # get random seed 
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    try:
        # choose tabs select
        if tabs == 0:
            template_images = eval(selected_template_images)
        elif tabs == 1:
            template_images = [init_image]
        elif tabs == 2:
            template_images = [file_d['name'] for file_d in uploaded_template_images]
        elif tabs == 3:
            pass
    except Exception as e:
        torch.cuda.empty_cache()
        return "Please choose or upload a template.", [], []
    
    # create modelscope model
    if retinaface_detection is None:
        retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface', model_revision='v2.0.2')
    if image_face_fusion is None:
        image_face_fusion       = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo', model_revision='v1.3')
    if face_skin is None:
        face_skin               = Face_Skin(os.path.join(models_path, "Others", "face_skin.pth"))
    if skin_retouching is None:
        try:
            skin_retouching     = pipeline('skin-retouching-torch', model='damo/cv_unet_skin_retouching_torch', model_revision='v1.0.2')
        except Exception as e:
            torch.cuda.empty_cache()
            logging.error(f"Skin Retouching model load error. Error Info: {e}")
    if portrait_enhancement is None:
        try:
            portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement', model_revision='v1.0.0')
        except Exception as e:
            torch.cuda.empty_cache()
            logging.error(f"Portrait Enhancement model load error. Error Info: {e}")
    
    # To save the GPU memory, create the face recognition model for computing FaceID if the user intend to show it.
    if display_score and face_recognition is None:
        face_recognition = pipeline("face_recognition", model='bubbliiiing/cv_retinafce_recognition', model_revision='v1.0.3')

    # params init
    input_prompts                   = []
    face_id_images                  = []
    roop_images                     = []
    face_id_retinaface_boxes        = []
    face_id_retinaface_keypoints    = []
    face_id_retinaface_masks        = []
    sd_lora_checkpoints             = []
    input_prompt_without_lora       = additional_prompt
    multi_user_facecrop_ratio       = 1.5
    multi_user_safecrop_ratio       = 1.0
    # Second diffusion hr scale
    default_hr_scale                = 1.0
    need_mouth_fix                  = True

    sd_model_checkpoint             = os.path.join(models_path, f"Stable-diffusion/{sd_model_checkpoint}")
    sd_xl_model_checkpoint          = os.path.join(models_path, f"Stable-diffusion/{SDXL_MODEL_NAME}")

    logging.info("Start templates and user_ids preprocess.")
    for user_id in user_ids:
        if user_id == 'none':
            # use some placeholder 
            input_prompts.append('none')
            face_id_images.append('none')
            roop_images.append('none')
            face_id_retinaface_boxes.append([])
            face_id_retinaface_keypoints.append([])
            face_id_retinaface_masks.append([])
        else:
            # get prompt
            input_prompt            = f"{validation_prompt}" + additional_prompt
            
            # get best image after training
            best_outputs_paths = glob.glob(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", "*.jpg"))
            # get roop image
            if len(best_outputs_paths) > 0:
                face_id_image_path  = best_outputs_paths[0]
            else:
                face_id_image_path  = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg") 
            roop_image_path         = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

            sd_lora_checkpoint = [os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", f"{user_id}.safetensors")]
            if os.path.exists(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", f"ddpo_{user_id}.safetensors")):
                logging.info("Exist DDPO weight, load it")
                sd_lora_checkpoint.append(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", f"ddpo_{user_id}.safetensors"))

            face_id_image           = Image.open(face_id_image_path).convert("RGB")
            roop_image              = Image.open(roop_image_path).convert("RGB")

            # Crop user images to obtain portrait boxes, facial keypoints, and masks
            _face_id_retinaface_boxes, _face_id_retinaface_keypoints, _face_id_retinaface_masks = call_face_crop(retinaface_detection, face_id_image, multi_user_facecrop_ratio, "face_id")
            _face_id_retinaface_box      = _face_id_retinaface_boxes[0]
            _face_id_retinaface_keypoint = _face_id_retinaface_keypoints[0]
            _face_id_retinaface_mask     = _face_id_retinaface_masks[0]

            input_prompts.append(input_prompt)
            face_id_images.append(face_id_image)
            roop_images.append(roop_image)
            face_id_retinaface_boxes.append(_face_id_retinaface_box)
            face_id_retinaface_keypoints.append(_face_id_retinaface_keypoint)
            face_id_retinaface_masks.append(_face_id_retinaface_mask)
            sd_lora_checkpoints.append(sd_lora_checkpoint)

    if tabs == 3:
        logging.info(sd_xl_input_prompt)
        sd_xl_resolution = eval(str(sd_xl_resolution))
        template_images = sdxl_txt2img(
            input_prompt = sd_xl_input_prompt, \
            diffusion_steps=30, width=sd_xl_resolution[1], height=sd_xl_resolution[0], \
            default_positive_prompt=DEFAULT_POSITIVE_XL, \
            default_negative_prompt=DEFAULT_NEGATIVE_XL, \
            seed=seed, sd_model_checkpoint=sd_xl_model_checkpoint, 
        )
        template_images = [np.uint8(template_images)]

    outputs, face_id_outputs    = [], []
    loop_message                = ""
    for template_idx, template_image in enumerate(template_images):
        template_idx_info = f'''
            Start Generate template                 : {str(template_idx + 1)};
            user_ids                                : {str(user_ids)};
            input_prompts                           : {str(input_prompts)};
            before_face_fusion_ratio                : {str(before_face_fusion_ratio)}; 
            after_face_fusion_ratio                 : {str(after_face_fusion_ratio)};
            first_diffusion_steps                   : {str(first_diffusion_steps)}; 
            first_denoising_strength                : {str(first_denoising_strength)}; 
            second_diffusion_steps                  : {str(second_diffusion_steps)};
            second_denoising_strength               : {str(second_denoising_strength)};
            seed                                    : {str(seed)}
            crop_face_preprocess                    : {str(crop_face_preprocess)}
            apply_face_fusion_before                : {str(apply_face_fusion_before)}
            apply_face_fusion_after                 : {str(apply_face_fusion_after)}
            color_shift_middle                      : {str(color_shift_middle)}
            color_shift_last                        : {str(color_shift_last)}
            super_resolution                        : {str(super_resolution)}
            display_score                           : {str(display_score)}
            background_restore                      : {str(background_restore)}
            background_restore_denoising_strength   : {str(background_restore_denoising_strength)}
        '''
        logging.info(template_idx_info)
        try:
            # open the template image
            if tabs == 0 or tabs == 2:
                template_image = Image.open(template_image).convert("RGB")
            else:
                template_image = Image.fromarray(template_image).convert("RGB")

            template_face_safe_boxes, _, _ = call_face_crop(retinaface_detection, template_image, multi_user_safecrop_ratio, "crop")
            if len(template_face_safe_boxes) == 0:
                return "Please upload a template with face.", [], []
            template_detected_facenum = len(template_face_safe_boxes)
            
            # use some print/log to record mismatch of detectionface and user_ids
            if template_detected_facenum > len(user_ids) - len(passed_userid_list):
                logging.info(f"User set {len(user_ids) - len(passed_userid_list)} face but detected {template_detected_facenum} face in template image,\
                the last {template_detected_facenum-len(user_ids) - len(passed_userid_list)} face will remains")
            
            if len(user_ids) - len(passed_userid_list) > template_detected_facenum:
                logging.info(f"User set {len(user_ids) - len(passed_userid_list)} face but detected {template_detected_facenum} face in template image,\
                the last {len(user_ids) - len(passed_userid_list)-template_detected_facenum} set user_ids is useless")

            if background_restore:
                output_image = np.array(copy.deepcopy(template_image))
                output_mask  = np.ones_like(output_image) * 255

                for index in range(len(template_face_safe_boxes)):
                    retinaface_box = template_face_safe_boxes[index]
                    output_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 0
                output_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(output_mask), np.ones((32, 32), np.uint8), iterations=1)))
            else:
                if min(template_detected_facenum, len(user_ids) - len(passed_userid_list)) > 1:
                    output_image = np.array(copy.deepcopy(template_image))
                    output_mask  = np.ones_like(output_image)

                    # get mask in final diffusion for multi people
                    for index in range(len(template_face_safe_boxes)):
                        # pass this userid, not mask the face
                        if index in passed_userid_list:
                            continue
                        else:
                            retinaface_box = template_face_safe_boxes[index]
                            output_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
                    output_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(output_mask), np.ones((64, 64), np.uint8), iterations=1) - cv2.erode(np.array(output_mask), np.ones((32, 32), np.uint8), iterations=1)))

            total_processed_person = 0
            for index in range(min(len(template_face_safe_boxes), len(user_ids) - len(passed_userid_list))):
                # pass this userid, not do anything
                if index in passed_userid_list:
                    continue
                total_processed_person += 1

                loop_template_image = copy.deepcopy(template_image)

                # mask other people face use 255 in this term, to transfer multi user to single user situation
                if min(len(template_face_safe_boxes), len(user_ids) - len(passed_userid_list)) > 1:
                    loop_template_image = np.array(loop_template_image)
                    for sub_index in range(len(template_face_safe_boxes)):
                        if index != sub_index:
                            retinaface_box = template_face_safe_boxes[sub_index]
                            loop_template_image[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
                    loop_template_image = Image.fromarray(np.uint8(loop_template_image))

                # Crop the template image to retain only the portion of the portrait
                if crop_face_preprocess:
                    loop_template_crop_safe_boxes, _, _ = call_face_crop(retinaface_detection, loop_template_image, 3, "crop")
                    loop_template_crop_safe_box = loop_template_crop_safe_boxes[0]
                    input_image = copy.deepcopy(loop_template_image).crop(loop_template_crop_safe_box)
                else:
                    input_image = copy.deepcopy(loop_template_image)

                # Resize the template image with short edges on 512
                logging.info("Start Image resize to 512.")
                short_side  = min(input_image.width, input_image.height)
                resize      = float(short_side / 512.0)
                new_size    = (int(input_image.width//resize), int(input_image.height//resize))
                input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
                if crop_face_preprocess:
                    new_width   = int(np.shape(input_image)[1] // 32 * 32)
                    new_height  = int(np.shape(input_image)[0] // 32 * 32)
                    input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
                
                # Detect the box where the face of the template image is located and obtain its corresponding small mask
                logging.info("Start face detect.")
                input_image_retinaface_boxes, input_image_retinaface_keypoints, input_masks = call_face_crop(retinaface_detection, input_image, 1.05, "template")
                input_image_retinaface_box      = input_image_retinaface_boxes[0]
                input_image_retinaface_keypoint = input_image_retinaface_keypoints[0]
                input_mask                      = input_masks[0]

                # backup input template and mask
                origin_input_mask               = copy.deepcopy(input_mask)
                original_input_template         = copy.deepcopy(input_image)

                # Paste user images onto template images
                replaced_input_image = crop_and_paste(face_id_images[index], face_id_retinaface_masks[index], input_image, face_id_retinaface_keypoints[index], input_image_retinaface_keypoint, face_id_retinaface_boxes[index])
                replaced_input_image = Image.fromarray(np.uint8(replaced_input_image))
                
                # Fusion of user reference images and input images as canny input
                if roop_images[index] is not None and apply_face_fusion_before:
                    fusion_image = image_face_fusion(dict(template=input_image, user=roop_images[index]))[OutputKeys.OUTPUT_IMG]
                    fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))

                    # The edge shadows generated by fusion are filtered out by taking intersections of masks of faces before and after fusion.
                    # detect face area
                    fusion_image_mask = np.int32(np.float32(face_skin(fusion_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 12, 13]])[0]) > 128)
                    input_image_mask = np.int32(np.float32(face_skin(input_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 12, 13]])[0]) > 128)
                    # paste back to photo
                    fusion_image = fusion_image * fusion_image_mask * input_image_mask + np.array(input_image) * (1 - fusion_image_mask * input_image_mask)
                    fusion_image = cv2.medianBlur(np.uint8(fusion_image), 3)
                    fusion_image = Image.fromarray(fusion_image)
                    
                    input_image = Image.fromarray(np.uint8((np.array(input_image, np.float32) * (1 - before_face_fusion_ratio) + np.array(fusion_image, np.float32) * before_face_fusion_ratio)))

                # Expand the template image in the x-axis direction to include the ears.
                h, w, c     = np.shape(input_mask)
                input_mask  = np.zeros_like(np.array(input_mask, np.uint8))
                input_image_retinaface_box = np.int32(input_image_retinaface_box)

                face_width                      = input_image_retinaface_box[2] - input_image_retinaface_box[0]
                input_image_retinaface_box[0]   = np.clip(np.array(input_image_retinaface_box[0], np.int32) - face_width * 0.10, 0, w - 1)
                input_image_retinaface_box[2]   = np.clip(np.array(input_image_retinaface_box[2], np.int32) + face_width * 0.10, 0, w - 1)

                # get new input_mask
                input_mask[input_image_retinaface_box[1]:input_image_retinaface_box[3], input_image_retinaface_box[0]:input_image_retinaface_box[2]] = 255
                input_mask = Image.fromarray(np.uint8(input_mask))
                
                # here we get the retinaface_box, we should use this Input box and face pixel to refine the output face pixel colors
                template_image_original_face_area = np.array(original_input_template)[input_image_retinaface_box[1]:input_image_retinaface_box[3], input_image_retinaface_box[0]:input_image_retinaface_box[2], :] 
                
                # First diffusion, facial reconstruction
                logging.info("Start First diffusion.")
                controlnet_pairs = [["canny", input_image, 0.50], ["openpose", replaced_input_image, 0.50], ["color", input_image, 0.85]]
                first_diffusion_output_image = inpaint(input_image, input_mask, controlnet_pairs, diffusion_steps=first_diffusion_steps, denoising_strength=first_denoising_strength, input_prompt=input_prompts[index], hr_scale=1.0, seed=str(seed), sd_model_checkpoint=sd_model_checkpoint, sd_lora_checkpoint=sd_lora_checkpoints[index])

                if color_shift_middle:
                    # apply color shift
                    logging.info("Start color shift middle.")
                    first_diffusion_output_image_uint8 = np.uint8(np.array(first_diffusion_output_image))
                    # crop image first
                    first_diffusion_output_image_crop = Image.fromarray(first_diffusion_output_image_uint8[input_image_retinaface_box[1]:input_image_retinaface_box[3], input_image_retinaface_box[0]:input_image_retinaface_box[2],:])
                    
                    # apply color shift
                    first_diffusion_output_image_crop_color_shift = np.array(copy.deepcopy(first_diffusion_output_image_crop))
                    first_diffusion_output_image_crop_color_shift = color_transfer(first_diffusion_output_image_crop_color_shift, template_image_original_face_area)
                    
                    # detect face area
                    face_skin_mask = np.int32(np.float32(face_skin(first_diffusion_output_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 12, 13]])[0]) > 128)
                    # paste back to photo
                    first_diffusion_output_image_uint8[input_image_retinaface_box[1]:input_image_retinaface_box[3], input_image_retinaface_box[0]:input_image_retinaface_box[2],:] = \
                        first_diffusion_output_image_crop_color_shift * face_skin_mask + np.array(first_diffusion_output_image_crop) * (1 - face_skin_mask)
                    first_diffusion_output_image = Image.fromarray(np.uint8(first_diffusion_output_image_uint8))

                # Second diffusion
                if roop_images[index] is not None and apply_face_fusion_after:
                    # Fusion of facial photos with user photos
                    logging.info("Start second face fusion.")
                    fusion_image = image_face_fusion(dict(template=first_diffusion_output_image, user=roop_images[index]))[OutputKeys.OUTPUT_IMG] # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                    fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
                    
                    # The edge shadows generated by fusion are filtered out by taking intersections of masks of faces before and after fusion.
                    # detect face area
                    fusion_image_mask = np.int32(np.float32(face_skin(fusion_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 12, 13]])[0]) > 128)
                    input_image_mask = np.int32(np.float32(face_skin(first_diffusion_output_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 12, 13]])[0]) > 128)
                    # paste back to photo
                    fusion_image = fusion_image * fusion_image_mask * input_image_mask + np.array(first_diffusion_output_image) * (1 - fusion_image_mask * input_image_mask)
                    fusion_image = cv2.medianBlur(np.uint8(fusion_image), 3)
                    fusion_image = Image.fromarray(fusion_image)

                    input_image = Image.fromarray(np.uint8((np.array(first_diffusion_output_image, np.float32) * (1 - after_face_fusion_ratio) + np.array(fusion_image, np.float32) * after_face_fusion_ratio)))

                else:
                    fusion_image = first_diffusion_output_image
                    input_image = first_diffusion_output_image

                # Add mouth_mask to avoid some fault lips, close if you dont need
                if need_mouth_fix:
                    logging.info("Start mouth detect.")
                    mouth_mask, face_mask = face_skin(input_image, retinaface_detection, [[4, 5, 12, 13], [1, 2, 3, 4, 5, 10, 12, 13]])
                    # Obtain the mask of the area around the face
                    face_mask = Image.fromarray(np.uint8(cv2.dilate(np.array(face_mask), np.ones((32, 32), np.uint8), iterations=1) - cv2.erode(np.array(face_mask), np.ones((16, 16), np.uint8), iterations=1)))

                    i_h, i_w, i_c = np.shape(face_mask)
                    m_h, m_w, m_c = np.shape(mouth_mask)
                    if i_h != m_h or i_w != m_w:
                        face_mask = face_mask.resize([m_w, m_h])
                    input_mask = Image.fromarray(np.uint8(np.clip(np.float32(face_mask) + np.float32(mouth_mask), 0, 255)))
                
                logging.info("Start Second diffusion.")
                controlnet_pairs = [["canny", fusion_image, 1.00], ["tile", fusion_image, 1.00]]
                second_diffusion_output_image = inpaint(input_image, input_mask, controlnet_pairs, input_prompts[index], diffusion_steps=second_diffusion_steps, denoising_strength=second_denoising_strength, hr_scale=default_hr_scale, seed=str(seed), sd_model_checkpoint=sd_model_checkpoint, sd_lora_checkpoint=sd_lora_checkpoints[index])

                # use original template face area to shift generated face color at last
                if color_shift_last:
                    logging.info("Start color shift last.")
                    # scale box
                    rescale_retinaface_box = [int(i * default_hr_scale) for i in input_image_retinaface_box]
                    second_diffusion_output_image_uint8 = np.uint8(np.array(second_diffusion_output_image))
                    second_diffusion_output_image_crop = Image.fromarray(second_diffusion_output_image_uint8[rescale_retinaface_box[1]:rescale_retinaface_box[3], rescale_retinaface_box[0]:rescale_retinaface_box[2],:])

                    # apply color shift
                    second_diffusion_output_image_crop_color_shift = np.array(copy.deepcopy(second_diffusion_output_image_crop)) 
                    second_diffusion_output_image_crop_color_shift = color_transfer(second_diffusion_output_image_crop_color_shift, template_image_original_face_area)

                    # detect face area
                    face_skin_mask = np.int32(np.float32(face_skin(second_diffusion_output_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10]])[0]) > 128)
                    # paste back to photo
                    second_diffusion_output_image_uint8[rescale_retinaface_box[1]:rescale_retinaface_box[3], rescale_retinaface_box[0]:rescale_retinaface_box[2],:] = \
                        second_diffusion_output_image_crop_color_shift * face_skin_mask + np.array(second_diffusion_output_image_crop) * (1 - face_skin_mask)
                    second_diffusion_output_image = Image.fromarray(second_diffusion_output_image_uint8)
                    
                # If it is a large template for cutting, paste the reconstructed image back
                if crop_face_preprocess:
                    logging.info("Start paste crop image to origin template.")
                    origin_loop_template_image      = np.array(copy.deepcopy(loop_template_image))

                    x1,y1,x2,y2                     = loop_template_crop_safe_box
                    second_diffusion_output_image   = second_diffusion_output_image.resize([x2-x1, y2-y1], Image.Resampling.LANCZOS)
                    origin_loop_template_image[y1:y2,x1:x2] = np.array(second_diffusion_output_image) 

                    loop_output_image               = Image.fromarray(np.uint8(origin_loop_template_image))
                else:
                    loop_output_image               = second_diffusion_output_image
                
                # Given the current user id, compute the FaceID of the second diffusion generation w.r.t the roop image.
                # For simplicity, we don't compute the FaceID of the final output image.
                if display_score:
                    loop_output_image = np.array(loop_output_image)
                    x1, y1, x2, y2 = loop_template_crop_safe_box
                    loop_output_image_face = loop_output_image[y1:y2, x1:x2]

                    embedding = face_recognition(dict(user=Image.fromarray(np.uint8(loop_output_image_face))))[OutputKeys.IMG_EMBEDDING]
                    roop_image_embedding = face_recognition(dict(user=Image.fromarray(np.uint8(roop_images[index]))))[OutputKeys.IMG_EMBEDDING]
                    
                    loop_output_image_faceid = np.dot(embedding, np.transpose(roop_image_embedding))[0][0]
                    # Truncate the user id to ensure the full information showing in the Gradio Gallery.
                    face_id_outputs.append((roop_images[index], "{}, {:.2f}".format(user_ids[index][:10], loop_output_image_faceid)))
                    loop_output_image = Image.fromarray(loop_output_image)
                
                if min(len(template_face_safe_boxes), len(user_ids) - len(passed_userid_list)) > 1:
                    logging.info("Start paste crop image to origin template in multi people.")
                    template_face_safe_box = template_face_safe_boxes[index]
                    output_image[template_face_safe_box[1]:template_face_safe_box[3], template_face_safe_box[0]:template_face_safe_box[2]] = np.array(loop_output_image, np.float32)[template_face_safe_box[1]:template_face_safe_box[3], template_face_safe_box[0]:template_face_safe_box[2]]
                else:
                    output_image = loop_output_image 

            try:
                if min(len(template_face_safe_boxes), len(user_ids) - len(passed_userid_list)) > 1 or background_restore:
                    logging.info("Start Thirt diffusion for background.")
                    output_image    = Image.fromarray(np.uint8(output_image))
                    short_side      = min(output_image.width, output_image.height)
                    if output_image.width / output_image.height > 1.5 or output_image.height / output_image.width > 1.5:
                        target_short_side = 512
                    else:
                        target_short_side = 768
                    resize          = float(short_side / target_short_side)
                    new_size        = (int(output_image.width//resize), int(output_image.height//resize))
                    output_image    = output_image.resize(new_size, Image.Resampling.LANCZOS)
                    # When reconstructing the entire background, use smaller denoise values with larger diffusion_steps to prevent discordant scenes and image collapse.
                    denoising_strength  = background_restore_denoising_strength if background_restore else 0.3
                    controlnet_pairs    = [["canny", output_image, 1.00], ["color", output_image, 1.00]]
                    output_image    = inpaint(output_image, output_mask, controlnet_pairs, input_prompt_without_lora, 30, denoising_strength=denoising_strength, hr_scale=1, seed=str(seed), sd_model_checkpoint=sd_model_checkpoint)
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Background Restore Failed, Please check the ratio of height and width in template. Error Info: {e}")
                return f"Background Restore Failed, Please check the ratio of height and width in template. Error Info: {e}", outputs, []
            
            try:
                logging.info("Start Skin Retouching.")
                # Skin Retouching is performed here. 
                output_image = Image.fromarray(cv2.cvtColor(skin_retouching(output_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Skin Retouching error: {e}")

            try:
                logging.info("Start Portrait enhancement.")
                h, w, c = np.shape(np.array(output_image))
                # Super-resolution is performed here. 
                if super_resolution:
                    output_image = Image.fromarray(cv2.cvtColor(portrait_enhancement(output_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Portrait enhancement error: {e}")

            if total_processed_person == 0:
                output_image = template_image
            else:
                outputs.append(output_image)
            save_image(output_image, easyphoto_outpath_samples)

            if loop_message != "":
                loop_message += "\n"
            loop_message += f"Template {str(template_idx + 1)} Success."
        except Exception as e:
            torch.cuda.empty_cache()
            logging.error(f"Template {str(template_idx + 1)} error: Error info is {e}, skip it.")

            if loop_message != "":
                loop_message += "\n"
            loop_message += f"Template {str(template_idx + 1)} error: Error info is {e}."

    torch.cuda.empty_cache()
    return "Success", outputs, face_id_outputs  