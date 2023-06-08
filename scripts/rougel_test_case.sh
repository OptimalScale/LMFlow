#!/bin/bash

# 比较self-instruct和LMFlow对相同instructions的ROUGE-L得分是否相同

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/test_rougel.py \
    --answer_type text \
    --model_name_or_path gpt2-large \
    --dataset_path data/alpaca/test \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric rouge-l