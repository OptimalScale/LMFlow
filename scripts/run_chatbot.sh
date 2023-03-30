#!/bin/bash

model=gpt2
if [ $# -ge 1 ]; then
  model=$1
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/chatbot.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model}
