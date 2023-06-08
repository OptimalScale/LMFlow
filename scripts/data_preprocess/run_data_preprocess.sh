#!/bin/bash
# Run this shell script under project directory

# For sample.py
python scripts/data_preprocess/sample.py \
    --dataset_path ./data/example_dataset/train/train_50.json \
    --output_path ./data/example_dataset/train/train_50_sample.json \
    --ratio 0.5

# For shuffle.py
python scripts/data_preprocess/shuffle.py \
    --dataset_path ./data/example_dataset/train/train_50_sample.json \
    --output_path ./data/example_dataset/train/train_50_sample_shuffle.json

# For merge.py : you can specify multiple files to merge
python scripts/data_preprocess/merge.py \
    --merge_from_path ./data/example_dataset/train/train_50_sample_shuffle.json \
    ./data/example_dataset/train/train_50_sample.json ./data/example_dataset/train/train_50.json \
    --output_path ./data/example_dataset/train/train_merge.json \

# For merge.py: if you want to merge a directory
python scripts/data_preprocess/merge.py \
    --merge_from_path ./data/example_dataset/train/*.json \
    --output_path ./data/example_dataset/train/train_merge.json \