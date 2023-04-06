#!/bin/bash

model=eachadea/vicuna-13b
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  deepspeed examples/chatbot.py \
      --use_ram_optimized_load False \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      ${lora_args}
