accelerate launch --config_file=./accelerate_t2i_config.yaml \
    finetune_t2i.py \
    --model_name_or_path="stabilityai/stable-diffusion-2-1" \
    --model_type="unet" \
    --use_lora=True \
    --lora_target_module "to_k" "to_q" "to_v" "to_out.0" "add_k_proj" "add_v_proj" \
    --dataset_path="./data/example" \
    --image_folder="img" \
    --image_size=768 \
    --train_file="train.json" \
    --validation_file="valid.json" \
    --output_dir="output" \
    --logging_dir="logs" \
    --overwrite_output_dir=True \
    --mixed_precision="fp16" \
    --num_train_epochs=100 \
    --train_batch_size=1 \
    --learning_rate=1e-4 \
    --valid_steps=50
