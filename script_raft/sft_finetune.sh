#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11040"      # Default argument
exp_id=raft_fintune
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/xx

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path $1 \
    --dataset_path $3 \
    --output_dir $2 --overwrite_output_dir \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 888888 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing True
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
