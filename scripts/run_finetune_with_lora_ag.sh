#!/bin/bash
# Please run this script under ${project_id} in project directory of

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 2 ]; then
  deepspeed_args="$2"
fi
if [ $# -ge 3 ]; then
  finetune_args="$3"
fi

exp_id=$1
output_dir=output_models/${exp_id}
log_dir=log/${exp_id}

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --output_dir ${output_dir} --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --use_lora 1 \
    --save_aggregated_lora 1\
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name finetune_with_lora \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
