#!/bin/bash

# Parses arguments
model_name_or_path=gpt2
lora_model_path=output_models/lora
output_model_path=output_models/merge_lora

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--lora_model_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

python examples/merge_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --lora_model_path ${lora_model_path} \
    --output_model_path ${output_model_path} \
    --deepspeed configs/ds_config_eval.json
