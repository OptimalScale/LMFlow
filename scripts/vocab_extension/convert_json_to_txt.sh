#!/bin/bash

bash data/download.sh wiki_zh_eval

python utils/convert_json_to_txt.py --dataset_path ./data/wiki_zh_eval \
        --output_path ./data/wiki_zh_eval/converted_data.txt \
        --overwrite True