export MODEL_NAME="Linaqruf/animagine-xl"
export OUTPUT_DIR="lora/arcaea-xl-1.1"
export HUB_MODEL_ID="sheriyuo/arcaea-xl-lora"
export DATASET_NAME="sheriyuo/arcaea"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=1500 \
  --validation_prompt="arcaea" \
  --seed=1337 \
  --variant="fp16" \
  --mixed_precision="fp16" \
  --push_to_hub \
  --hub_token="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"