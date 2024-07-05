#!/bin/bash
# Please run this script under ${project_id} in project directory of

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=multistage_finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}
dataset_path="${project_dir}/data/example_dataset/train"
if [ ! -d ${dataset_path} ]; then
  cd data && ./download.sh example_dataset && cd -
fi

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/multistage_finetune.py \
    --num_stages_per_epoch 1 \
    --run_name ${exp_id} \
    --model_name_or_path facebook/galactica-1.3b \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 3 \
    --learning_rate 1e-3 \
    --block_size 512 \
    --per_device_train_batch_size 2 \
    --use_lora 1 \
    --lora_r 8 \
    --save_aggregated_lora 1 \
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
