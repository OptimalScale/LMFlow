#!/bin/bash
mkdir -p ./output_models/merged_tokenizer
python utils/train_tokenizer.py --dataset_path ./data/wiki_zh_eval/converted_data.txt \
        --model_type bpe \
        --output_dir ./output_models/new_tokenizer \
        --user_defined_symbols 0,1,2,3,4,5,6,7,8,9,% \
        --vocab_size 20000 \
        --max_sentencepiece_length 4