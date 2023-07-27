#!/bin/bash

project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/llama-13b-catcontribution-10

CUDA_VISIBLE_DEVICES='' python scripts_for_pruning_study/prune_layers.py \
    --model_name_or_path pinkmanlove/llama-13b-hf\
    --torch_dtype bfloat16 \
    --layers_to_be_pruned "4,10,11,32,33,3,5,7,8,9" \
    --output_model_path ${output_dir}
    