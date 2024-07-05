#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh

if [ ! -d data/alpaca ]; then
  cd data && ./download.sh alpaca && cd -
fi

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path facebook/galactica-1.3b \
    --lora_model_path output_models/finetune_with_lora \
    --dataset_path data/alpaca/test \
    --prompt_structure "Input: {input}" \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy
