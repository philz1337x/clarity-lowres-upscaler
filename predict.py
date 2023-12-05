import os
import shutil
import torch
import settings
from PIL import Image
from typing import Iterator
from cog import BasePredictor, Input, Path
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline
)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        print("Loading controlnet tiles...")
        self.controlnet_tiles = ControlNetModel.from_pretrained(
            os.path.join(settings.MODEL_CACHE, "tiles"),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        print("Loading tiles_img2img...")
        self.tiles_img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            os.path.join(settings.MODEL_CACHE, "tiles_img2img"),
            controlnet=self.controlnet_tiles,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker = None,
        ).to('cuda')

        self.tiles_img2img.scheduler.config.algorithm_type = 'sde-dpmsolver++'
        self.tiles_img2img.scheduler = DPMSolverMultistepScheduler.from_config(self.tiles_img2img.scheduler.config, use_karras_sigmas=True)

    def load_image(self, image_path: Path):
        if image_path is None:
            return None
        if os.path.exists("img.png"):
            os.unlink("img.png")
        shutil.copy(image_path, "img.png")
    
        img = Image.open("img.png")
        return img
    
    def resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / max(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Source image", default=None
        ),
        output_size: int = Input(
            description="Size of output image", default=1024
        ),
        prompt: str = Input(
            description="Input prompt",
            default="best quality",
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
        ),
        strength: float = Input(
            description="Strength",
            default=1.0,
        ),
        steps: int = Input(
            description="Number of steps",
            default=32,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        source_image = self.load_image(image)

        condition_image = self.resize_for_condition_image(source_image, output_size)

        #torch.cuda.empty_cache() 
        
        this_seed = seed

        image_a = self.tiles_img2img(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            image=condition_image, 
            control_image=condition_image, 
            width=condition_image.size[0],
            height=condition_image.size[1],
            strength=strength,
            generator=torch.manual_seed(this_seed),
            num_inference_steps=steps,
        ).images[0]

        output_path = Path(f"/tmp/seed-{this_seed}-a.png")
        image_a.save(output_path)
        yield Path(output_path)