import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers import StableDiffusionUpscalePipeline, DPMSolverMultistepScheduler

from utils import get_mask
from control_nets import get_mlsd_image

def set_img_dims(img, min_dim=1024):
    """
    Resizes the image so that its smallest dimension is equal to min_dim while 
    maintaining the aspect ratio.
    
    Args:
        img (PIL.Image): The input image.
        min_dim (int): The minimum dimension size (default is 1024).

    Returns:
        PIL.Image: The resized image.
    """
    w, h = img.size
    # Calculate scaling factor to make the smallest dimension at least min_dim
    if min(w, h) < min_dim:
        scaler = min_dim / min(w, h)
        img = img.resize((int(w * scaler), int(h * scaler)), Image.ANTIALIAS)
    return img


def resize_image(img):
    """
    Resizes the image such that both its dimensions are multiples of 8, which is 
    often required for certain neural network architectures.
    
    Args:
        img (PIL.Image): The input image.

    Returns:
        PIL.Image: The resized image.
    """
    original_width, original_height = img.size

    # Calculate the new dimensions
    new_width = original_width - (original_width % 8)
    new_height = original_height - (original_height % 8)

    # Resize the image while maintaining the aspect ratio
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img



class Staging:
    def __init__(self, control_net="mlsd", device="cuda"):
        """
        Initializes the Staging class with a Stable Diffusion inpainting pipeline, 
        scheduler, and random generator.

        Args:
            seed (bool): Whether to set a random seed (default is False).
            roomtype (str): The type of room (default is "bedroom").
            device (str): The device to run the pipeline on (default is "cuda").
        """
        
        # controlnet = ControlNetModel.from_pretrained("checkpoints/checkpoint-500/controlnet", torch_dtype=torch.float16)
        controlnet = ControlNetModel.from_pretrained("checkpoints/controlnet", torch_dtype=torch.float16)
        
        self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "checkpoints/juggerxlInpaint_juggerInpaintV8.safetensors",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        
        
        # controlnet = ControlNetModel.from_pretrained(
        #     "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
        #     )
        
        # self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        #     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        #     # "/home/nash/stable-diffusion-webui-forge/models/Stable-diffusion/juggerxlInpaint_juggerInpaintV8.safetensors",
        #     controlnet=controlnet,
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        # ).to(device)

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, use_karras_sigmas=True
        )
        self.pipeline.safety_checker = None
        
    def load_lora(self, room_type):
        self.pipeline.unload_lora_weights()

        self.pipeline.load_lora_weights('checkpoints/add-detail-xl.safetensors', weights=1.2)
        print("Loaded add-detail")
                
        if room_type == "bedroom":
            self.pipeline.load_lora_weights('checkpoints/bedroom.safetensors')
            print("Loaded bedroom-LoRA")


    def load_controlnet(self, controlnet):

        self.pipeline.controlnet_map = None
        torch.cuda.empty_cache()
                
        if controlnet == "mlsd":
            controlnet = ControlNetModel.from_pretrained("checkpoints/checkpoint-500/controlnet", torch_dtype=torch.float16)
            print("Loaded mlsd-controlnet")


    def predict(self, prompt, negative_prompt, image, mask_image):
        """
        Generates an inpainted image based on the given prompt and mask.

        Args:
            prompt (str): The text prompt for the inpainting model.
            negative_prompt (str): The negative prompt for the inpainting model.
            image (PIL.Image): The input image.
            mask_image (PIL.Image): The mask image indicating the region to be inpainted.

        Returns:
            PIL.Image: The inpainted output image.
        """
        width, height = image.size
        out_img = self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask=mask_image,
            num_inference_steps=30,
            num_images_per_prompt=1,
            preprocessed=True,
            height=height,
            width=width,
        ).images[0]
        return out_img

    def upscale_img(self, prompt, inp_img, num_steps=30):
        """
        Upscales the input image based on the given prompt.

        Args:
            prompt (str): The text prompt for the upscaling model.
            inp_img (PIL.Image): The input image to be upscaled.
            num_steps (int): The number of inference steps (default is 30).

        Returns:
            PIL.Image: The upscaled output image.
        """
        return self.upscale_pipeline(
            prompt=prompt, image=inp_img, num_inference_steps=num_steps
        ).images[0]

    def preprocess_img(self, img, blur_factor=5, padding_factor=5):
        """
        Preprocesses the input image by resizing it and generating a mask.

        Args:
            img (PIL.Image): The input image.
            blur_factor (int): The blur factor for the mask (default is 5).

        Returns:
            tuple: The preprocessed image and the corresponding mask.
        """
        img = set_img_dims(img)
        img = resize_image(img)
        mask = np.array(get_mask(img, padding_factor))
        mask = mask.astype("uint8")
        mask = self.pipeline.mask_processor.blur(
            Image.fromarray(mask), blur_factor=blur_factor
        )
        return img, mask

    def __call__(self, prompt, negative_prompt, image, preprocessed=False, mask=None, **kwargs):
        """
        Calls the inpainting pipeline with the given parameters.

        Args:
            prompt (str): The text prompt for the inpainting model.
            negative_prompt (str): The negative prompt for the inpainting model.
            image (PIL.Image): The input image.
            preprocessed (bool): Whether the image has already been preprocessed (default is False).
            mask (PIL.Image): The mask image indicating the region to be inpainted (default is None).

        Returns:
            Output: The output from the inpainting pipeline.
        """
        if not preprocessed:
            print("preprocessing image...")
            padding_factor = kwargs["padding_factor"]
            blur_factor = kwargs["blur_factor"]
            image, mask = self.preprocess_img(image, blur_factor, padding_factor)
        
        # Removing extra keys
        kwargs.pop('padding_factor', None)
        kwargs.pop('blur_factor', None)
        kwargs.pop('blur_factor', None)
        
        # Load appropriate room_type LoRA
        room_type = kwargs.pop('room_type')
        # self.load_lora(room_type)
        
        # Check if a controlnet is selected
        if kwargs.get("controlnet"):
            kwargs.pop('controlnet')
            control_image = get_mlsd_image(image)
        
        if (kwargs.get("width") is None) or (kwargs.get("width") == 0):
            kwargs["width"] = image.size[0]
        if (kwargs.get("height") is None) or (kwargs.get("height") == 0):
            kwargs["height"] = image.size[1]

        out_imgs = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            control_image=control_image,
            **kwargs
        )
        return out_imgs, mask, control_image
