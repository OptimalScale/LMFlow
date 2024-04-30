#!/bin/bash

# Parses arguments
model_name_or_path=gpt2
lora_model_path=output_models/lora
output_model_path=output_models/merge_lora
device=cpu

# if gpu
deepspeed_args="--master_port=11000"

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    --model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    --lora_model_path)
      lora_model_path="$2"
      shift
      ;;
    --output_model_path)
      output_model_path="$2"
      shift
      ;;
    --device)
      device="$2"
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


if [ ${device} == "cpu" ]; then
    python examples/merge_lora.py \
        --model_name_or_path ${model_name_or_path} \
        --lora_model_path ${lora_model_path} \
        --output_model_path ${output_model_path} \
        --device ${device} \
        --ds_config configs/ds_config_eval.json
elif [ ${device} == "gpu" ]; then
    echo "Error: Merging LoRA weights using gpu not supported yet. Please use cpu."
else
    echo "Error: Unknown device \"${device}\"" 1>&2
    exit 1
fi
