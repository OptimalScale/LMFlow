#!/bin/bash

# download data
cd data && bash download.sh wiki_zh_eval && cd -

# convert json to txt for sentencepiece
python utils/convert_json_to_txt.py --dataset_path ./data/wiki_zh_eval \
        --output_path ./data/wiki_zh_eval/converted_data.txt \
        --overwrite True

# train a new tokenizer
mkdir -p ./output_models/new_tokenizer
python utils/train_tokenizer.py --dataset_path ./data/wiki_zh_eval/converted_data.txt \
        --model_type bpe \
        --output_dir ./output_models/new_tokenizer \
        --user_defined_symbols 0,1,2,3,4,5,6,7,8,9,% \
        --vocab_size 20000 \
        --max_sentencepiece_length 4

# merge the new tokenizer with the old one
mkdir -p ./output_models/merged_tokenizer
python utils/merge_tokenizer.py --chinese_sp_model_file ./output_models/new_tokenizer/example.model \
        --tokenizer_dir openlm-research/open_llama_3b \
        --output_dir ./output_models/merged_tokenizer