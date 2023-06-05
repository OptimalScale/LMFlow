#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text \
    --model_name_or_path gpt2-large \
    --dataset_path data/alpaca/test \
    --deepspeed examples/ds_config.json \
    --metric accuracy