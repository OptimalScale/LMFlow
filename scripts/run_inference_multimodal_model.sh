#!/bin/bash

model="Salesforce/blip-image-captioning-base"
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/inference.py \
      --deepspeed configs/ds_config_multimodal.json \
      --model_name_or_path ${model} \
      --arch_type vision_encoder_decoder \
      ${lora_args}
