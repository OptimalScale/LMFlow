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
    --dataset_path ./data/example_dataset/train/train_50.json \
    --merge_from_path ./data/example_dataset/train/train_50_sample_shuffle.json \
    ./data/example_dataset/train/train_50_sample.json  \
    --output_path ./data/example_dataset/train/train_merge.json \

# For concat.py: if you simply want to merge multiple files or a directory, use following.
# You can also specify multiple files after --merge_from_path
python scripts/data_preprocess/concat.py \
    --merge_from_path ./data/example_dataset/train/*.json \
    --output_path ./data/example_dataset/train/train_merge.json \

# For concat_shuffle_split.py: if you simply want to merge multiple files or a directory, use following.
python scripts/data_preprocess/concat_shuffle_split.py \
    --merge_from_path ./data/example_dataset/train/*.json \
    --output_path ./data/processed_dataset/ \