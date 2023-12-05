import os
import shutil
import settings
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

TMP_CACHE = "tmp_cache"

if os.path.exists(TMP_CACHE):
    shutil.rmtree(TMP_CACHE)
os.makedirs(TMP_CACHE)

cn_tiles = ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL_TILES,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
# cn_tiles.half()
cn_tiles.save_pretrained(os.path.join(settings.MODEL_CACHE, 'tiles'))

tiles_img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    settings.REAL_BASE_MODEL,
    controlnet=cn_tiles,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
    safety_checker = None,
).to('cuda')

tiles_img2img.save_pretrained(os.path.join(settings.MODEL_CACHE, 'tiles_img2img'))

shutil.rmtree(TMP_CACHE)