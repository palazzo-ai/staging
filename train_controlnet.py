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
#  --validation_prompt "modern looking bedroom mirror" "cyan circle with brown floral background" \
#  --validation_steps=100 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --seed=42 \
#  --report_to="wandb" \
#  --push_to_hub


# accelerate launch train_controlnet_sdxl.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --dataset_name=khurramkhalil/mlsd_sdxl \
#  --mixed_precision="fp16" \
#  --resolution=1024 \
#  --learning_rate=1e-5 \
#  --max_train_steps=150000 \
#  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
#  --validation_prompt "The image showcases a modern living room with a green accent wall, a white fireplace, a brown leather sofa, a wooden coffee table, and various decorative elements such as potted plants, vases, and a television." "The image showcases a cozy, well-organized living room with a white bookshelf filled with various books, a small statue of a cat, and a framed photo, all contributing to a warm and inviting atmosphere." \
#  --validation_steps=100 \
#  --train_batch_size=1 \
#  --checkpointing_steps=5000 \
#  --gradient_accumulation_steps=4 \
#  --report_to="wandb" \
#  --seed=42 \
#  --push_to_hub
