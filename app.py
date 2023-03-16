import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("Lykon/DreamShaper")

pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_attention_slicing()
with autocast("cuda"):
    image = pipe(prompt).images[0]  
    print(image)
    image.save("image.png")
