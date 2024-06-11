#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
# Parses arguments
model_name_or_path=/home/yizhenjia/.cache/huggingface/hub/models--EleutherAI--pythia-1b-deduped/snapshots/7199d8fc61a6d565cd1f3c62bf11525b563e13b2
reward_model_name_or_path=/home/yizhenjia/.cache/huggingface/hub/models--EleutherAI--pythia-1b-deduped/snapshots/7199d8fc61a6d565cd1f3c62bf11525b563e13b2
train_dataset_path=/vol/yizhenjia/projs/LMFlow/data/alpaca/train_conversation
output_dir=output_models/ppo
conversation_template=gemma

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
exp_id=ppo
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml \
    examples/ppo.py \
        --model_name_or_path ${model_name_or_path} \
        --reward_model_name_or_path ${reward_model_name_or_path} \
        --do_train True \
        --do_eval True \
        --dataset_path ${train_dataset_path} \
        --conversation_template ${conversation_template} \
        --output_dir ${output_dir} --overwrite_output_dir \
        --use_flash_attention True \
        --block_size 64 \
        --learning_rate 1e-5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 0.01 \
        --num_ppo_epochs 1 \
        --gradient_accumulation_steps 32 \
        --report_to 'wandb' \
        --run_name ${exp_id} \
        --preprocessing_num_workers 4 \
        | tee ${log_dir}/train.log \
        2> ${log_dir}/train.err