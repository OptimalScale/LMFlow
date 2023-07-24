#!/bin/bash

python examples/train_tokenizer.py --dataset_path ./data/wiki_zh_eval/converted_data.txt \
        --output_dir ./output_models/new_tokenizer \
        --model_type bpe \
        --vocab_size 10000 \
        --user_defined_symbols 0,1,2,3,4,5,6,7,8,9,%