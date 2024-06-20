import os
import random
from PIL import Image

import gc
import torch


# Import the Staging class from the model.staging module
from model.staging import Staging
from utils import set_img_dims
from prompts import RANDOM_PHRASES

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# Instantiate the Staging model
model = Staging()

# Open the input image
# image = Image.open("test_images/1.jpg")
room_types = ["bedroom", "living room"]
architecture_styles = ["modern", "scandinavian", "boho", "industrial", "contemporary"]

def staging(idx, image, room_type, architecture_style):
    # Optionally, you can include pre-processing checks
    # pre_process_check = is_img_good(image)

    # Define the room type and architectural style for the prompt
    # room_type = random.choice(room_types)
    # architecture_style = "modern"

    # Create the text prompt for the model
    # prompt = f"The (({architecture_style} {room_type})), style, ((best quality)),((masterpiece)),((realistic))"
    prompt = "coastal living room, ultra-realistic, 4K, 8K, HD, high quality, photorealistic, professional, highly detailed, real life, high-resolution, full HD, high-resolution image, 8K Ultra HD,  high detailed,  hyperdetailed photography"
    # prompt += random.choice(RANDOM_PHRASES)
    negative_prompt = "blurry, unrealistic, synthatic, window, door, fireplace, out of order, deformed, disfigured, watermark, text, banner, logo, contactinfo, surreal longbody, lowres, bad anatomy, bad hands, jpeg artifacts, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, rug"

    # Set parameters for the model inference
    room_type = "bedroom"
    num_inference_steps = 20
    num_images_per_prompt = 2
    strength = 0.7
    seed = -1
    guidance_scale = 5
    padding_factor = 5
    blur_factor = 5
    print(prompt)
    # Generate output images using the model
    output_images, mask = model(
        prompt,
        negative_prompt=negative_prompt,
        image=image,
        room_type=room_type,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_inference_step=30,
        seed=seed,
        padding_factor=padding_factor,
        blur_factor=blur_factor
    )

    # Save the generated images to the output directory
    for idxx, output in enumerate(output_images.images):
        output.save(f"test_output/{idx+1}_image.jpg")
    mask.save(f"test_output/{idx+1}_mask.jpg")
    flush()

dir = "test"
img_list = os.listdir(dir)

idx = 3
image = Image.open("test/4.jpg")
image = set_img_dims(image)
room_type = "bedroom"
architecture_style = "architecture_style"
staging(idx, image, room_type, architecture_style)

# for idx, item in enumerate(img_list):
#     # Open the input image
#     image = Image.open(f"{dir}/" + item)
#     image = set_img_dims(image)

#     for room_type in room_types:
#         for architecture_style in architecture_styles:
#             print(item)

#     staging(idx, image, room_type, architecture_style)
