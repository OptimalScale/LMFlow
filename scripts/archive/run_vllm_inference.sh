#!/bin/bash
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.

# Parses arguments
run_name=vllm_inference
model_name_or_path='Qwen/Qwen2-0.5B'
dataset_path=data/alpaca/test_conversation
output_dir=data/inference_results
output_file_name=results.json
apply_chat_template=True

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
    --output_dir)
      output_dir="$2"
      shift
      ;;
    --output_file_name)
      output_file_name="$2"
      shift
      ;;
    --apply_chat_template)
      apply_chat_template="$2"
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

python examples/vllm_inference.py \
  --use_vllm True \
  --trust_remote_code ${trust_remote_code} \
  --model_name_or_path ${model_name_or_path} \
  --dataset_path ${dataset_path} \
  --preprocessing_num_workers 16 \
  --random_seed 42 \
  --apply_chat_template ${apply_chat_template} \
  --num_output_sequences 2 \
  --use_beam_search False \
  --temperature 1.0 \
  --top_p 0.9 \
  --max_new_tokens 1024 \
  --save_results True \
  --results_path ${output_file_path} \
  --enable_decode_inference_result False \
  --vllm_gpu_memory_utilization 0.95 \
  --vllm_tensor_parallel_size 2 \
  --enable_distributed_vllm_inference False \
  2>&1 | tee ${log_dir}/vllm_inference.log