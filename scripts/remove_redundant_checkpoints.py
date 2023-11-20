#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=self-assigning-variable
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=literal-comparison

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pathlib
import shutil
import sys
import time

from datetime import datetime


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
        formatter_class=argparse.RawTextHelpFormatter
    ) 

    # Add arguments
    parser.add_argument(
        '--checkpoint_dir', type=str, required=True, default=None,
        help='directory with structure {checkpoint_dir}/checkpoint-*/*'
    )
    parser.add_argument(
        '--save_total_limit', type=int, default=11,
        help='maximum number of existing checkpoints',
    )
    parser.add_argument(
        '--save_frequency', type=int, default=1000,
        help='checkpoint-K with K % {save_frequency} == 0 will be retained',
    )
    parser.add_argument(
        '--sleep_time', type=int, default=-1,
        help='how frequently check the directory, -1 means check only once',
    )
    parser.add_argument(
        '--debug_mode', type=int, default=0,
        help='when this value is 1, only simulation results will be printed',
    )

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def sub(list_a, list_b):
    return list(set(list_a) - set(list_b))


def remove_redundant_checkpoints(args):
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    # First scan for counting
    checkpoint_id_list = []
    for checkpoint in checkpoint_dir.iterdir():
        if not checkpoint.is_dir():
            continue

        try:
            checkpoint_id = int(checkpoint.name.lstrip('checkpoint-'))
        except:
            continue

        checkpoint_id_list.append(checkpoint_id)

    # Retains K % save_frequency == 0 checkpoints
    retain_id_list = []
    num_remain_slots = args.save_total_limit
    for checkpoint_id in reversed(sorted(checkpoint_id_list)):
        if num_remain_slots <= 0:
            break
        if checkpoint_id % args.save_frequency == 0:
            retain_id_list.append(checkpoint_id)
            num_remain_slots -= 1
            continue

    checkpoint_id_list = sub(checkpoint_id_list, retain_id_list)

    # If there is still space, retain latest checkpoints
    num_remain_slots = args.save_total_limit - len(retain_id_list)
    if num_remain_slots > 0:
        latest_id_list = sorted(checkpoint_id_list)[-num_remain_slots:]
        retain_id_list += latest_id_list
        checkpoint_id_list = sub(checkpoint_id_list, latest_id_list)

    # Removes 
    remove_id_list = checkpoint_id_list
    if args.debug_mode:
        for checkpoint_id in sorted(remove_id_list):
            checkpoint_path = checkpoint_dir / f'checkpoint-{checkpoint_id}'
            print('-', checkpoint_path)

        for checkpoint_id in sorted(retain_id_list):
            checkpoint_path = checkpoint_dir / f'checkpoint-{checkpoint_id}'
            print('+', checkpoint_path)
    else:
        for checkpoint_id in sorted(remove_id_list):
            checkpoint_path = checkpoint_dir / f'checkpoint-{checkpoint_id}'
            shutil.rmtree(checkpoint_path)


def main():
    """Removes redundant checkpoints"""
    args = parse_argument(sys.argv)
    print('#################################################')
    print('args =', str(args))

    if args.sleep_time < 0:
        remove_redundant_checkpoints(args)
    else:
        while True:
            remove_redundant_checkpoints(args)
            time.sleep(args.sleep_time)


if __name__ == '__main__':
    main()
