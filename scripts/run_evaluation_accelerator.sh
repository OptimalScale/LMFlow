#!/bin/bash

if [ ! -d data/MedQA-USMLE ]; then
  cd data && ./download.sh MedQA-USMLE && cd -
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerator_singlegpu_config.yaml examples/evaluation.py \
    --answer_type usmle \
    --model_name_or_path gpt2-large \
    --dataset_path data/MedQA-USMLE/validation \
    --use_ram_optimized_load True \
    --deepspeed examples/ds_config.json \
    --metric accuracy \
    --output_dir output_dir/accelerator_1_card \
    --inference_batch_size_per_device 1 \
    --use_accelerator_for_evaluator True \
    --torch_dtype bfloat16
