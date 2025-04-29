#!/bin/bash
model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
dataset_path=/home/yizhenjia/datasets/10_doubled
conversation_template=qwen2_5
output_dir=/home/yizhenjia/models/scalebio/qwen1.5b/10-doubled-unif
export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT='scalebio'

# Finetune
exp_id=qwen1.5b-10-doubled-unif
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
