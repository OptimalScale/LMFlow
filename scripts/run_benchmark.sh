#!/bin/bash

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  help_message="./$(basename $0)"
  help_message+=" --dataset_name DATASET_NAME"
  help_message+=" --model_name_or_path MODEL_NAME_OR_PATH"
  echo ${help_message} 1>&2
  exit 1
fi

extra_args="--dataset_name gpt4_en_eval --model_name_or_path gpt2"
if [ $# -ge 1 ]; then
  extra_args="$@"
fi


CUDA_VISIBLE_DEVICES=0 \
  deepspeed --master_port 11001 examples/benchmarking.py \
  --use_ram_optimized_load 0 \
  --deepspeed examples/ds_config.json \
  --metric nll \
  --prompt_structure "###Human: {input}###Assistant:" \
  ${extra_args} 