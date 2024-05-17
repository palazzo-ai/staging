# from accelerate.utils import write_basic_config
# write_basic_config()

# from datasets import load_dataset

# dataset = load_dataset("fusing/fill50k")

# export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
# export OUTPUT_DIR="circle-controlnet-sdxl"

# accelerate launch train_controlnet_sdxl.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --dataset_name=fusing/fill50k \
#  --mixed_precision="fp16" \
#  --resolution=1024 \
#  --learning_rate=1e-5 \
#  --max_train_steps=15000 \
#  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
#  --validation_steps=100 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --seed=42 \
#  --report_to="wandb" \
#  --push_to_hub