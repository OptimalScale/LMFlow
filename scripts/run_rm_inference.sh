#!/bin/bash
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.

# Parses arguments
run_name=rm_inference
model_name_or_path=sfairXC/FsfairX-LLaMA3-RM-v0.1
dataset_path=data/alpaca/test
output_dir=data/rm_inference_results
output_file_name=results.json
conversation_template=llama3

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -r|--run_name)
      run_name="$2"
      shift
      ;;
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    --output_dir)
      output_dir="$2"
      shift
      ;;
    --output_file_name)
      output_file_name="$2"
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

# inference
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${run_name}
output_file_path=${output_dir}/${run_name}/${output_file_name}
mkdir -p ${output_dir}/${run_name} ${log_dir}

accelerate launch --config_file configs/accelerator_multigpu_config.yaml \
    examples/rm_inference.py \
        --trust_remote_code ${trust_remote_code} \
        --model_name_or_path ${model_name_or_path} \
        --arch_type text_regression \
        --use_accelerator True \
        --block_size 4096 \
        --inference_batch_size 16 \
        --dataset_path ${dataset_path} \
        --overwrite_cache True \
        --conversation_template ${conversation_template} \
        --preprocessing_num_workers 16 \
        --save_results True \
        --results_path ${output_file_path} \
        2>&1 | tee ${log_dir}/rm_inference.log