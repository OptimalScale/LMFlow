#!/usr/bin/env python
# coding=utf-8

import argparse
import logging

import json
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data/wiki_zh_eval', type=str, required=False)
    parser.add_argument('--output_path', default='./data/wiki_zh_eval/converted_data.txt', type=str, required=False)
    parser.add_argument('--overwrite', default=False, type=bool, required=False)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    outputfile = args.output_path

    outputs_list = []
    data_files = [
                    x.absolute().as_posix()
                    for x in Path(dataset_path).glob("*.json")
                ]

    for file_name in data_files:
        with open(file_name) as fin:
            json_data = json.load(fin)
            type = json_data["type"]
            for line in json_data["instances"]:
                outputs_list.append(line["text"])
                

    if Path(outputfile).exists() and not args.overwrite:
        logging.warning(f"File %s exists, will not overwrite.", outputfile)
    else:
        with open(outputfile, "w") as f:
            for line in outputs_list:
                f.write(line)

