'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import cv2
import numpy as np
import concurrent.futures

import torch
from model.staging import Staging

from diffusers.utils import load_image
from utils import set_img_dims

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

# from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #

MODEL = Staging()

# ---------------------------------- Helper ---------------------------------- #

def decode_base64_image(image_string):
    image_binary = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(image_binary))
    return image


def image_to_base64(output_image, ext=".jpg"):
    _, encoded_image = cv2.imencode(ext, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    image_string = base64.b64encode(encoded_image).decode("utf-8")
    return image_string


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls



@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    if job_input.get('image_url'):
        starting_image = job_input['image_url']
        image = load_image(starting_image).convert("RGB")
    else:
        starting_image = job_input['image']
        image = decode_base64_image(starting_image)

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    
    image = set_img_dims(image)
    
    prompt = job_input.get("prompt", None)
    negative_prompt = job_input.get("negative_prompt", None)
    num_images_per_prompt = job_input.get("num_images", 1)
    num_inference_step = job_input.get("num_inference_step", 30)
    guidance_scale = job_input.get("guidance_scale", 5)
    seed = job_input.get("seed", -1)
    width = job_input.get('width', None)
    height = job_input.get('height', None)
    
    padding_factor = job_input.get('mask_padding', 5)
    blur_factor = job_input.get('blur_factor', 5)
    
    output, mask = MODEL(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_images_per_prompt=1,
        num_inference_step=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        padding_factor=padding_factor,
        blur_factor=blur_factor
    )

    # image_urls = _save_and_upload_images(output, job['id'])
    image_string = []
    for img in output.images:
        if not isinstance(img, np.ndarray):
            output_image = np.array(img)
        image_string.append(image_to_base64(output_image))

    if not isinstance(mask, np.ndarray):
        mask_image = np.array(mask)
    mask_image = image_to_base64(mask_image)
    
    
    results = {
        "result": image_string[0],
        "mask": mask_image,
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results


runpod.serverless.start({"handler": generate_image})