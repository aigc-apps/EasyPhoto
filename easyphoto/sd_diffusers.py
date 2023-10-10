import logging
import os
import re
from collections import defaultdict

import torch
import torch.utils.checkpoint
from diffusers import (DPMSolverMultistepScheduler,
                       StableDiffusionControlNetInpaintPipeline,
                       StableDiffusionXLPipeline)
from easyphoto.easyphoto_config import preload_lora
from easyphoto.train_kohya.utils.model_utils import \
    load_models_from_stable_diffusion_checkpoint
from safetensors.torch import load_file
from transformers import CLIPTokenizer

tokenizer       = None
scheduler       = None
text_encoder    = None
vae             = None
unet            = None
pipeline        = None
sd_model_checkpoint_before  = ""
weight_dtype                = torch.float16
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

def merge_lora(pipeline, lora_path, multiplier, from_safetensor=False, device='cpu', dtype=torch.float32):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if from_safetensor:
        state_dict = load_file(lora_path, device=device)
    else:
        checkpoint = torch.load(os.path.join(lora_path, 'pytorch_lora_weights.bin'), map_location=torch.device(device))
        new_dict = dict()
        for idx, key in enumerate(checkpoint):
            new_key = re.sub(r'\.processor\.', '_', key)
            new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
            new_key = re.sub('_lora.up.', '.lora_up.', new_key)
            new_key = re.sub('_lora.down.', '.lora_down.', new_key)
            new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
            new_key = re.sub('to_out', 'to_out_0', new_key)
            new_key = 'lora_unet_' + new_key
            new_dict[new_key] = checkpoint[key]
            state_dict = new_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                    weight_down.squeeze(3).squeeze(2)).unsqueeze(
                2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

# TODO: Refactor with merge_lora.
def unmerge_lora(pipeline, lora_path, multiplier=1, from_safetensor=False, device="cpu", dtype=torch.float32):
    """Unmerge state_dict in LoRANetwork from the pipeline in diffusers."""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if from_safetensor:
        state_dict = load_file(lora_path, device=device)
    else:
        checkpoint = torch.load(os.path.join(lora_path, 'pytorch_lora_weights.bin'), map_location=torch.device(device))
        new_dict = dict()
        for idx, key in enumerate(checkpoint):
            new_key = re.sub(r'\.processor\.', '_', key)
            new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
            new_key = re.sub('_lora.up.', '.lora_up.', new_key)
            new_key = re.sub('_lora.down.', '.lora_down.', new_key)
            new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
            new_key = re.sub('to_out', 'to_out_0', new_key)
            new_key = 'lora_unet_' + new_key
            new_dict[new_key] = checkpoint[key]
            state_dict = new_dict

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                    weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

def t2i_sdxl_call(
    steps=20,
    seed=-1,

    cfg_scale=7.0,
    width=640,
    height=768,

    prompt="",
    negative_prompt="",
    sd_model_checkpoint="",
):  
    width   = int(width // 8 * 8)
    height  = int(height // 8 * 8)

    # Load scheduler, tokenizer and models.
    sdxl_pipeline = StableDiffusionXLPipeline.from_single_file(sd_model_checkpoint).to("cuda", weight_dtype)
    sdxl_pipeline.scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012)

    try:
        import xformers
        sdxl_pipeline.enable_xformers_memory_efficient_attention()
    except:
        logging.warning('No module named xformers. Infer without using xformers. You can run pip install xformers to install it.')

    generator = torch.Generator("cuda").manual_seed(int(seed)) 

    image = sdxl_pipeline(
        prompt, negative_prompt=negative_prompt, 
        guidance_scale=cfg_scale, num_inference_steps=steps, generator=generator, height=height, width=width
    ).images[0]

    del sdxl_pipeline
    torch.cuda.empty_cache()
    return image

def i2i_inpaint_call(
    images=[],  
    mask_image=None,  
    denoising_strength=0.75,
    controlnet_image=[],
    controlnet_units_list=[],
    controlnet_conditioning_scale=[],
    steps=20,
    seed=-1,

    cfg_scale=7.0,
    width=640,
    height=768,

    prompt="",
    negative_prompt="",
    sd_lora_checkpoint=[],
    sd_model_checkpoint="",
    sd_base15_checkpoint="",
):  
    global tokenizer, scheduler, text_encoder, vae, unet, sd_model_checkpoint_before, pipeline
    width   = int(width // 8 * 8)
    height  = int(height // 8 * 8)
    
    if (sd_model_checkpoint_before != sd_model_checkpoint) or (unet is None) or (vae is None) or (text_encoder is None):
        sd_model_checkpoint_before = sd_model_checkpoint
        text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(False, sd_model_checkpoint)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_base15_checkpoint, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        sd_base15_checkpoint, subfolder="tokenizer"
    )

    pipeline = StableDiffusionControlNetInpaintPipeline(
        controlnet=controlnet_units_list, 
        unet=unet.to(weight_dtype),
        text_encoder=text_encoder.to(weight_dtype),
        vae=vae.to(weight_dtype),
        scheduler=noise_scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
    ).to("cuda")
    if preload_lora is not None:
        for _preload_lora in preload_lora:
            merge_lora(pipeline, _preload_lora, 0.60, from_safetensor=True, device="cuda", dtype=weight_dtype)
    if len(sd_lora_checkpoint) != 0:
        # Bind LoRANetwork to pipeline.
        for _sd_lora_checkpoint in sd_lora_checkpoint:
            merge_lora(pipeline, _sd_lora_checkpoint, 0.90, from_safetensor=True, device="cuda", dtype=weight_dtype)

    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        logging.warning('No module named xformers. Infer without using xformers. You can run pip install xformers to install it.')
        
    generator           = torch.Generator("cuda").manual_seed(int(seed)) 
    pipeline.safety_checker = None

    image = pipeline(
        prompt, image=images, mask_image=mask_image, control_image=controlnet_image, strength=denoising_strength, negative_prompt=negative_prompt, 
        guidance_scale=cfg_scale, num_inference_steps=steps, generator=generator, height=height, width=width, \
        controlnet_conditioning_scale=controlnet_conditioning_scale, guess_mode=True
    ).images[0]

    if len(sd_lora_checkpoint) != 0:
        # Bind LoRANetwork to pipeline.
        for _sd_lora_checkpoint in sd_lora_checkpoint:
            unmerge_lora(pipeline, _sd_lora_checkpoint, 0.90, from_safetensor=True, device="cuda", dtype=weight_dtype)
    if preload_lora is not None:
        for _preload_lora in preload_lora:
            unmerge_lora(pipeline, _preload_lora, 0.60, from_safetensor=True, device="cuda", dtype=weight_dtype)
    return image