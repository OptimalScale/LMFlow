#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Adds prompt structure to a text2text dataset.
"""
from __future__ import absolute_import

import argparse
import json
import textwrap
import sys

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
        help=textwrap.dedent("input dataset path, reads from stdin by default")
    )
    parser.add_argument(
        "--output_path", type=str,
        default=None,
        help=textwrap.dedent("output dataset path, writes to stdout by default")
    )
    parser.add_argument(
        "--end_mark", type=str,
        default="###",
        help=textwrap.dedent("end mark that append to the end of output")
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

    output_field_map = {
        "text_only": "text",
        "text2text": "output",
    }
    data_dict_type = data_dict["type"]
    if not data_dict_type in output_field_map:
        raise NotImplementedError(
            "only support text_only or text2text dataset"
        )

    output_field = output_field_map[data_dict_type]
    
    num_instances = len(data_dict["instances"])
    for i in range(num_instances):
        data_dict["instances"][i][output_field] += args.end_mark

    if args.output_path is not None:
        with open(args.output_path, "w") as fout:
            json.dump(data_dict, fout, indent=4, ensure_ascii=False)
    else:
        json.dump(data_dict, sys.stdout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
