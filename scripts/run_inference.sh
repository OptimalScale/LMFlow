#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:./src"

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/inference.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --test_file data/MedQA-USMLE/validation/valid_1273.json \
    --deepspeed examples/ds_config.json
