#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=13100"      # Default argument
# if [ $# -ge 1 ]; then
#   deepspeed_args="$1"
# fi

exp_id=pruning_study_wikitext/$1
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}


mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path $1 \
    --dataset_path /home/zhangyihan/projects/LMFlow/data/wikitext-103-raw-v1/train \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 0.5 \
    --learning_rate 1e-4 \
    --block_size 1024 \
    --per_device_train_batch_size 16 \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 888888 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1\
    --dataloader_num_workers 1 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 2 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err