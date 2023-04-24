#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Samples a certain ratio of instances from a dataset.
"""
from __future__ import absolute_import

import argparse
import json
import random
import sys
import textwrap

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
        "--dataset_path", type=str,
        default=None,
        help="input dataset path, reads from stdin by default"
    )
    parser.add_argument(
        "--output_path", type=str,
        default=None,
        help="output dataset path, writes to stdout by default"
    )
    parser.add_argument(
        "--ratio", type=float, required=True,
        help="sample ratio, will be floored if number of samples is not a int"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="pseudorandom seed"
    )

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def main():
    args = parse_argument(sys.argv)
    if args.dataset_path is not None:
        with open(args.dataset_path, "r") as fin:
            data_dict = json.load(fin)
    else:
        data_dict = json.load(sys.stdin)

    random.seed(args.seed)
    num_instances = len(data_dict["instances"])
    num_sample = int(num_instances * args.ratio)

    data_dict["instances"] = random.sample(data_dict["instances"], num_sample)

    if args.output_path is not None:
        with open(args.output_path, "w") as fout:
            json.dump(data_dict, fout, indent=4, ensure_ascii=False)
    else:
        json.dump(data_dict, sys.stdout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
