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
