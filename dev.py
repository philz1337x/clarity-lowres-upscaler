import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from diffusers.utils import load_image

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', 
                                             torch_dtype=torch.float16)
tiles_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("philz1337/epicrealism",
                                                                    controlnet=controlnet,
                                                                    torch_dtype=torch.float16).to('cuda')
tiles_pipe.enable_xformers_memory_efficient_attention()
tiles_pipe.enable_vae_slicing()
tiles_pipe.enable_vae_tiling()

source_image = load_image('https://replicate.delivery/pbxt/JzpU5yOIQOxj50y88m1zs9NnmWjzHTratVp5TEg3b4hpjEWE/GAfKBwPWkAAD8G4-2.jpeg')

condition_image = resize_for_condition_image(source_image, 1024)

torch.cuda.empty_cache() 

pipe = tiles_pipe(prompt="beautiful black woman, green flower dress", 
             negative_prompt="blurry, illustration, drawing, painting, over saturated, overexposed, tattoos, chocker, wig", 
             image=condition_image, 
             control_image=condition_image, 
             width=condition_image.size[0],
             height=condition_image.size[1],
             strength=1,
             generator=torch.manual_seed(0),
             num_inference_steps=50,
            ).images[0]

pipe.save('a.png')
