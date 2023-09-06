#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Converts a raw text file, separated by lines, into a "text-only" formatted json.
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

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def raw2textonly(fin):
    """
    Converts raw text to text-only format.

    Args:
        fin: the input file description of the raw text file.
    Returns:
        a dict with "text-only" format.
    """
    data_dict = {
        "type": "text_only",
        "instances": [ { "text": line.strip() } for line in fin ],
    }
    return data_dict


def main():
    args = parse_argument(sys.argv)

    if args.dataset_path is not None:
        with open(args.dataset_path, "r") as fin:
            data_dict = raw2textonly(fin)
    else:
        data_dict = raw2textonly(sys.stdin)

    if args.output_path is not None:
        with open(args.output_path, "w") as fout:
            json.dump(data_dict, fout, indent=4, ensure_ascii=False)
    else:
        json.dump(data_dict, sys.stdout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
