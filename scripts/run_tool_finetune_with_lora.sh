#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=gpt2
dataset_path=data/alpaca/train
output_dir=output_models/finetune
deepspeed_args="--master_port=12000"

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_lora_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune_with_lora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/tool_finetune_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 0.01 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --use_lora 0 \
    --lora_r 8 \
    --save_aggregated_lora 0\
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --lora_target_modules q_proj k_proj v_proj o_proj  \
    --trust_remote_code 1 \
    --tool_conv_template tool-llama-single-round \
    --lazy_preprocess True \
    --report_to none \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
