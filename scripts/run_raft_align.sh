#!/bin/bash
# Please run this script under project directory.

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=raft_align
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/imdb/train

mkdir -p ${output_dir} ${log_dir}

export PYTHONPATH=.
deepspeed ${deepspeed_args} \
    examples/raft_align.py \
    --model_name_or_path gpt2 \
    --num_train_epochs 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "constant" \
    --bf16 \
    --deepspeed configs/ds_config_zero2.json \
    --run_name ${exp_id} \
    --dataset_path ${dataset_path} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --output_dir ${output_dir} --overwrite_output_dir \
    --ddp_timeout 72000 \
    --save_steps 35 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 12 \
    | tee ${log_dir}/raft_align.log \
    2> ${log_dir}/raft_align.err
