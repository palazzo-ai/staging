## Installation
1. Create a New Conda Environment:
   ```bash
   conda create --n staging python=3.11
3. Activate conda enviroment
    ```bash
    conda activate staging
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
## Usage
To run the main script, execute the following command:
    ```bash
    python main.py
## Checkpoints
It is important to have checkpoints folder with SDXL checkpoint and add_detail LoRA safetensor.
## Project Structure
    main.py: Main script for running the project.
    model/: Directory containing the pretrained model initialization and calls script.
    test_images/: Directory containing test images.
    output/: Directory where generated images will be saved.
    utils.py: Utility functions.
    prompts.py: Definitions of prompts used for model inference.

## Parameters and Default Values

### Mandatory fields (Any one of the following)
   - `image_url = url`
   - `image = base64`

### Default values
   - `prompt = None`
   - `negative_prompt = None`
   - `num_images_per_prompt = 1`
   - `num_inference_step = 30`
   - `guidance_scale = 5`
   - `seed = -1`
   - `width = None`
   - `height = None`
   - `padding_factor = 5`
   - `blur_factor = 5`
   - `room_type = "bedroom"`

## Generic Request Format

```json
{
  "input": {
    "prompt": "The ((modern bedroom)), style, ((best quality)),((masterpiece)),((realistic))",
    "negative_prompt": "blurry, unrealistic, synthatic, window, door, fireplace, out of order, deformed, disfigured, watermark, text, banner, logo, contactinfo, surreal longbody, lowres, bad anatomy, bad hands, jpeg artifacts, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, rug",
    "image_url": "https://storage.googleapis.com/generative-models-output/empty_room.jpg",
    "num_inference_step": 25,
    "guidance_scale": 5,
    "strength": 0.3,
    "seed": -1,
    "num_images": 1
  }
}
```
## FastAPI
Start the application
```bash
    uvicorn app:app --reload
```
cURL request
```json
curl -X 'POST' \
  'http://127.0.0.1:8000/process?image_url=https%3A%2F%2Fstorage.googleapis.com%2Fgenerative-models-output%2Fempty_room.jpg&room_type=bedroom' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "a bed room",
  "negative_prompt": "string",
  "num_images_per_prompt": 1,
  "num_inference_step": 30,
  "guidance_scale": 5,
  "seed": -1,
  "width": 0,
  "height": 0,
  "padding_factor": 5,
  "blur_factor": 5
}'
