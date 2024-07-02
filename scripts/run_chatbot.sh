#!/bin/bash
# A simple chatbot script, the memory of the chatbot has a length of maximum
# model length, e.g. 4k for llama-2.

model=gpt2
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

    # --temperature 0.7 \
accelerate launch --config_file configs/accelerator_multigpu_config.yaml \
  examples/chatbot.py \
    --deepspeed configs/ds_config_chatbot.json \
    --model_name_or_path ${model} \
    --use_accelerator True \
    --max_new_tokens 256 \
    --temperature 1.0 \
    --end_string "#" \
    ${lora_args}
