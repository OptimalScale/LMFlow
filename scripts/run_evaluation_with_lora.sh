#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh
#for input in "Consider it carefully:" "give correct answer:" "Concise""responsible"
CUDA_VISIBLE_DEVICES=1 \
    deepspeed examples/evaluate.py \
    --answer_type usmle \
    --model_name_or_path pinkmanlove/llama-7b-hf \
    --lora_model_path output_models/llama7b-lora-medical \
    --dataset_path data/MedQA-USMLE/validation \
    --prompt_structure "Good job confident accurate:{input}" \
    --batch_size 30 \
    --deepspeed examples/ds_config.json \
#Which answer is the best?Please consider it carefully:
#confident