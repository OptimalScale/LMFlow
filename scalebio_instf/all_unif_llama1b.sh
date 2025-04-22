#!/bin/bash
export HF_TOKEN=''
model_name_or_path=/root/autodl-tmp/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
dataset_path=/root/autodl-tmp/projs/scalebio/data/all_no_val_test_fixed
conversation_template=llama3
output_dir=output_models/finetune/scalebio/instf/unif-all/llama1b

# Finetune
exp_id=finetune-scalebio-instf-unif-all-llama1b
log_dir=${output_dir}/log/
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_singlegpu_config.yaml \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code 1 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --disable_group_texts 1 \
    --num_train_epochs 1 \
    --block_size 2048 \
    --padding_side left \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
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
    --gradient_checkpointing 1 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --report_to wandb \
    --run_name ${exp_id} \
    --seed 42 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)
