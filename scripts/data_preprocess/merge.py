#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Merges an extra dataset into current dataset.
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
    
    parser.add_argument(
        "--dataset_path", type=str,
        default=None,
        help=textwrap.dedent("input dataset path, reads from stdin by default")
    )
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
    
    if args.merge_from_path is not None:
        for i in range(0, len(args.merge_from_path)):
            with open(args.merge_from_path[i], "r") as fin:
                extra_data_dict = json.load(fin)

            if data_dict["type"] != extra_data_dict["type"]:
                raise ValueError(
                    'two dataset have different types:'
                    f' input dataset: "{data_dict["type"]}";'
                    f' merge from dataset: "{extra_data_dict["type"]}"'
                )
            data_dict["instances"].extend(extra_data_dict["instances"])


    if args.output_path is not None:
        with open(args.output_path, "w") as fout:
            json.dump(data_dict, fout, indent=4, ensure_ascii=False)
    else:
        json.dump(data_dict, sys.stdout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
