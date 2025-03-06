#!/bin/bash
model_name_or_path=meta-llama/Llama-3.2-3B-Instruct
dataset_path=data/alpaca/train_conversation
conversation_template=llama3
output_dir=output_models/finetune_lora

# LoRA related arguments
lora_rank=8
lora_alpha=32
lora_dropout=0.1

# Finetune
exp_id=finetune_with_lora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code 0 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --use_lora 1 \
    --lora_r ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --disable_group_texts 1 \
    --num_train_epochs 1 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 \
    --torch_dtype bfloat16 \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --use_flash_attention 0 \
    --gradient_checkpointing 0 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --run_name ${exp_id} \
    --seed 42 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)
