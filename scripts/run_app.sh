#!/bin/bash

CUDA_VISIBLE_DEVICES=9 accelerate launch --config_file configs/accelerator_singlegpu_config.yaml service/app.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --torch_dtype bfloat16 \
    --use_int8 True \
    --max_new_tokens 200