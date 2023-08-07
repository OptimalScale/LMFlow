#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=rm
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/hh_rlhf/rm/hh_rlhf_rm_training.json
if [ ! -d data/hh_rlhf ]; then
  cd data && ./download.sh hh_rlhf && cd -
fi

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/reward_modeling.py \
    --model_name_or_path gpt2 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1\
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name rm_test \
    --validation_split_percentage 10 \
    --logging_steps 10 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 999999 \
    --evaluation_strategy steps\
    --eval_steps 100\
    --weight_decay 0.001\
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
