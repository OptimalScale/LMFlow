#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
# Parses arguments
model_name_or_path=google/gemma-2b-it
train_dataset_path=weqweasdas/preference_dataset_mix2
eval_dataset_path=weqweasdas/preference_dataset_mix2
output_dir=output_models/rewardmodeling
deepspeed_args="--master_port=11000"
conversation_template=llama2

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    --train_dataset_path)
      train_dataset_path="$2"
      shift
      ;;
    --eval_dataset_path)
      eval_dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=rewardmodeling
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
    examples/rewardmodeling.py \
        --deepspeed configs/ds_config_zero3.json \
        --model_name_or_path ${model_name_or_path} \
        --train_dataset_path ${train_dataset_path} \
        --eval_dataset_path ${eval_dataset_path} \
        --output_dir ${output_dir}  \
        --do_train True \
        --learning_rate 1e-5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 1 \
        --weight_decay 0.001 \
        --evaluation_strategy "steps" \
        --eval_steps 999999 \
        --save_strategy "steps" \
        --save_steps 999999 \
        --gradient_accumulation_steps 32 \
        --gradient_checkpointing True \
        --remove_unused_columns False \
        --bf16 True \
        --logging_strategy "steps" \
        --logging_steps 10 \
        --optim "paged_adamw_32bit" \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --report_to 'wandb' \
        | tee ${log_dir}/train.log \
        2> ${log_dir}/train.err