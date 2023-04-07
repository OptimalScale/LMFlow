#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type usmle \
    --model_name_or_path pinkmanlove/llama-7b-hf \
    --lora_model_path output_models/llama7b-lora-medical \
    --dataset_path data/MedQA-USMLE/validation \
    --prompt_structure "Input: {input}" \
    --batch_size 30 \
    --deepspeed examples/ds_config.json
