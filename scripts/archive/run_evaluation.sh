#!/bin/bash

if [ ! -d data/MedQA-USMLE ]; then
  cd data && ./download.sh MedQA-USMLE && cd -
fi

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --dataset_path data/MedQA-USMLE/validation \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy
