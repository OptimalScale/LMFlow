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

accelerate launch --config_file configs/accelerator_multigpu_config.yaml \
  examples/inference.py \
    --deepspeed configs/ds_config_chatbot.json \
    --model_name_or_path ${model} \
    --use_accelerator True \
    --max_new_tokens 256 \
    --temperature 1.0 \
    ${lora_args}
