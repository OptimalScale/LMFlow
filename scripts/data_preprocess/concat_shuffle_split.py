#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
This script is designed for handling large datasets. 
It merges multiple datasets located in the same directory, shuffles them, and splits them into training, evaluation, and testing sets. 
The training set is further divided into 10 folds.
"""
from __future__ import absolute_import

import argparse
import json
import textwrap
import sys
import os 
import random
import gc

def parse_argument(sys_argv):
    """Parses arguments from command line.
    Args:
        sys_argv: the list of arguments (strings) from command line.
    Returns:
        A struct whose member corresponds to the required (optional) variable.
        For example,
        ```
        args = parse_argument(['main.py' '--input', 'a.txt', '--num', '10'])
        args.input       # 'a.txt'
        args.num         # 10
        ```
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Training parameters
    parser.add_argument(
        "--output_path", type=str,
        default=None,
        help=textwrap.dedent("output dataset path, writes to stdout by default")
    )
    parser.add_argument(
        "--merge_from_path", type=str,
        nargs="+",
        help=textwrap.dedent(
            "dataset path of the extra dataset that will be merged"
            " into input dataset"
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help=textwrap.dedent("pseudorandom seed")
    )
    parser.add_argument(
        "--eval_size", type=int, default=200,
        help=textwrap.dedent("size of eval dataset")
    )
    parser.add_argument(
        "--test_size", type=int, default=1000,
        help=textwrap.dedent("size of test dataset")
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help=textwrap.dedent("the train dataset will be divide into k folds")
    )
    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def main():
    args = parse_argument(sys.argv)

    # concat 
    if args.merge_from_path is not None:
        for i in range(0, len(args.merge_from_path)):
            with open(args.merge_from_path[i], "r") as fin:
                extra_data_dict = json.load(fin)
            if i == 0:
                data_dict = extra_data_dict
            else:
                if data_dict["type"] != extra_data_dict["type"]:
                    raise ValueError(
                        'two dataset have different types:'
                        f' input dataset: "{data_dict["type"]}";'
                        f' merge from dataset: "{extra_data_dict["type"]}"'
                    )
                data_dict["instances"].extend(extra_data_dict["instances"])
    else:
        raise ValueError("No merge files specified")
    del extra_data_dict
    gc.collect()
    print('finish concat')

    # shuffle 
    random.seed(args.seed)
    random.shuffle(data_dict["instances"])
    print('finish shuffle')
    # split to train, eval, test
    train_data_dict = {"type":data_dict["type"],"instances":data_dict["instances"][args.eval_size:-args.test_size]}
    eval_data_dict = {"type":data_dict["type"],"instances":data_dict["instances"][:args.eval_size]}
    test_data_dict = {"type":data_dict["type"],"instances":data_dict["instances"][-args.test_size:]}
    del data_dict
    gc.collect()

    # divide train in 10 folds
    num_instances = len(train_data_dict["instances"])
    split_size = num_instances // args.k
    split_data = []
    for i in range(args.k):
        if i <  args.k-1:
            split = train_data_dict["instances"][i*split_size : (i+1)*split_size]
        else:
            # Last split may have remaining instances
            split = train_data_dict["instances"][i*split_size:]
        split_data.append({'type': train_data_dict["type"], 'instances': split})

    del train_data_dict
    gc.collect()

    print('finish split')
    # save dataset under output_path

    if args.output_path is  None:
        args.output_path = sys.stdout

    train_save_path=os.path.join(args.output_path,"train_{k}_folds".format(k=args.k))
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    for i in range(args.k):
        with open(train_save_path+"/train_"+str(i)+".json", 'w') as f:
            json.dump(split_data[i], f,  indent=4, ensure_ascii=False)

    eval_save_path=os.path.join(args.output_path,"eval")
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    with open(eval_save_path+'/eval.json','w') as f:
        json.dump(eval_data_dict,f,indent=4,ensure_ascii=False)

    test_save_path=os.path.join(args.output_path,"test")
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    with open(test_save_path+'/test.json','w') as f:
        json.dump(test_data_dict,f,indent=4,ensure_ascii=False)



if __name__ == "__main__":
    main()
