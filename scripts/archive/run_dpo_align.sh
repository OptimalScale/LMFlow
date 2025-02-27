#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=meta-llama/Llama-2-7b-hf
dataset_path=data/dpo-mix-7k
output_dir=output_models/dpo
deepspeed_args="--master_port=11000"
# specify gpus/single gpu here by 
# `--include localhost:0,1` or `--include localhost:0`

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
    -o|--output_dir)
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
exp_id=dpo
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/dpo_train.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    --run_name dpo \
    --max_steps 200 \
    --learning_rate 1e-6 \
    --use_lora 1 \
    --lora_r 8 \
    --sanity_check True \
    --save_aggregated_lora 0\
    --logging_steps 20 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
