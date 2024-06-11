from PIL import Image

import torch
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel


device = "cuda"
dtype = torch.float16

controlnet_mlsd = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-mlsd", torch_dtype=dtype, use_safetensors=True).to(device)

mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators").to(device)

def get_mlsd_image(img, resize=False):
    mlsd_img = mlsd_processor(img)
    
    if resize:
        mlsd_img = mlsd_img.resize((512, 512))
    
    return mlsd_img

if __name__ =="__main__":
    image = Image.open('image.jpg')
    mlsd_image = get_mlsd_image(image)
    # mlsd_image.save('controlnet.jpg')