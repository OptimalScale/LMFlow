import argparse
import os.path as osp
import torch
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert checkpoint from MiniGPT4")
    parser.add_argument("--data_path", type=str, help="the model path for the to convert checkpoint")
    parser.add_argument("--save_path", default=None, type=str, help="the save path for converted checkpoint")
    parser.add_argument("--max_length", default=1000, type=int, help="the max length for the text file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data = json.load(open(args.data_path))
    for data_idx in data:
        for item in data_idx['conversations']:
            if len(item["value"]) > args.max_length:
                item["value"] = item["value"][:args.max_length]
    with open(args.save_path, 'w') as f:
        json.dump(data, f)
    print("finish processing the data.")
            