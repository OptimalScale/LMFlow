#!/bin/bash
model_name_or_path=/mnt/yizhenjia3/ipt_final/trained_model/train_stage_5_of_5/checkpoint-9200
dataset_path=data/ifeval-like-default
conversation_template=qwen2_5
output_dir=output_models/finetune/ifeval-like-default-final-lora

# LoRA related arguments
lora_r=128
lora_alpha=256
lora_dropout=0.1

# Finetune
exp_id=finetune-ifeval-like-default-final-lora
log_dir=${output_dir}/log/
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code 1 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --use_lora 1 \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --disable_group_texts 1 \
    --num_train_epochs 1 \
    --block_size 4096 \
    --padding_side left \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 \
    --torch_dtype bfloat16 \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --use_flash_attention 1 \
    --gradient_checkpointing 0 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --report_to wandb \
    --run_name ${exp_id} \
    --seed 42 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)
