import os

# save_dirs
data_dir                        = "./"
data_path                       = data_dir

# models path
models_path                     = os.path.join(data_path, "model_data")
abs_models_path                 = os.path.abspath(models_path)

# java scripts
script_path                     = os.path.join(data_path, "javascript")

easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')

# prompts 
validation_prompt   = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE = 'beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
DEFAULT_NEGATIVE = 'glasses, naked, nude, nsfw, breasts, penis, cum,, ugly, huge eyes, text, logo, worst face, strange mouth, nsfw, NSFW, low quality, worst quality, worst quality, low quality, normal quality, lowres, watermark, lowres, monochrome, naked, nude, nsfw, bad anatomy, bad hands, normal quality, grayscale, mural,'

cache_log_file_path     = os.path.join(models_path, "train_kohya_log.txt")