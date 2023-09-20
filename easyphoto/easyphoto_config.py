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
DEFAULT_POSITIVE = 'best quality, realistic, photo-realistic, detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE = 'bags under the eyes:1.5, Bags under eyes, glasses:1.5, naked, nude, nsfw, breasts, penis, cum, worst quality, low quality, normal quality, over red lips, hair, teeth, lowres, watermark, badhand, normal quality, lowres, bad anatomy, bad hands, normal quality, mural,'

cache_log_file_path     = os.path.join(models_path, "train_kohya_log.txt")
preload_lora            = [os.path.join(models_path, "Lora/FilmVelvia3.safetensors")]