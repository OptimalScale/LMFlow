#!/bin/bash

model=gpt2
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/chatbot.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      ${lora_args}
