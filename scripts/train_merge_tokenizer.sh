#!/bin/bash

# download data
cd data
bash download.sh wiki_zh_eval
cd ..

# convert json to txt for sentencepiece
python scripts/data_preprocess/convert_json_to_txt.py --dataset_path ./data/wiki_zh_eval \
        --output_path ./data/wiki_zh_eval/converted_data.txt \
        --overwrite True

# train a new tokenizer
python examples/train_tokenizer.py --dataset_path ./data/wiki_zh_eval/converted_data.txt \
        --model_type bpe \
        --output_dir ./output_models/new_tokenizer \
        --user_defined_symbols 0,1,2,3,4,5,6,7,8,9,% \
        --vocab_size 10000

# merge the new tokenizer with the old one
python examples/merge_tokenizer.py --chinese_sp_model_file ./output_models/new_tokenizer/example.model \
        --tokenizer_dir pinkmanlove/llama-7b-hf \
        --output_dir ./output_models/merged_tokenizer