#!/bin/bash
mkdir -p ./output_models/new_tokenizer
python utils/merge_tokenizer.py --tokenizer_dir pinkmanlove/llama-7b-hf \
        --chinese_sp_model_file ./output_models/new_tokenizer/example.model \
        --output_dir ./output_models/merged_tokenizer \