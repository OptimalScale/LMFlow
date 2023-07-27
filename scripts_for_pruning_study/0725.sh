#!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh pinkmanlove/llama-13b-hf
CUDA_VISIBLE_DEVICES='1,2,3,4,5,6' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh output_models/llama-13b-meancontribution-10
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh output_models/llama-13b-catcontribution-10
