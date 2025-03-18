#!/bin/bash
# An interactive inference script without context history, i.e. the chatbot
# won't have conversation memory.

model=gpt2
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
  examples/inference.py \
    --model_name_or_path ${model} \
    --max_new_tokens 256 \
    --temperature 1.0 \
    ${lora_args}
