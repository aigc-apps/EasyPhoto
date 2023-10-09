import os

# save_dirs
data_dir                        = "./"
data_path                       = data_dir

# models path
models_path                     = os.path.join(data_path, "model_data")
abs_models_path                 = os.path.abspath(models_path)

# java scripts
script_path                     = os.path.join(data_path, "javascript")

# sample path
easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')

# log path and lora path
cache_log_file_path             = os.path.join(models_path, "train_kohya_log.txt")
preload_lora                    = [os.path.join(models_path, "Lora/FilmVelvia3.safetensors")]

# prompts 
validation_prompt   = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE    = 'cloth, best quality, realistic, photo-realistic, detailed skin, rough skin, beautiful eyes, sparkling eyes, beautiful mouth, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE    = 'bags under the eyes, bags under eyes, glasses, naked, nsfw, nude, breasts, penis, cum, over red lips, bad lips, bad hair, bad teeth, worst quality, low quality, normal quality, lowres, watermark, badhand, lowres, bad anatomy, bad hands, normal quality, mural,'
DEFAULT_POSITIVE_XL = 'film photography, a clear face, minor acne, high resolution detail of human skin texture, rough skin, indirect lighting'
DEFAULT_NEGATIVE_XL = 'nfsw, bokeh, cgi, illustration, cartoon, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, ugly, deformed, blurry, Noisy, log, text'

# ModelName
SDXL_MODEL_NAME     = 'SDXL_1.0_ArienMixXL_v2.0.safetensors'