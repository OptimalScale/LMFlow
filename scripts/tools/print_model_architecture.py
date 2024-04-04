#!/usr/bin/env python
#coding=utf-8
import argparse
import sys
from transformers import AutoModel

def parse_argument(sys_argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_name_or_path", type=str, default='gpt2')
    args = parser.parse_args(sys_argv[1:])
    return args

def main():
    args = parse_argument(sys.argv)
    model_name = args.model_name_or_path
    model = AutoModel.from_pretrained(model_name)

    print(model.config)
    print(model)

if __name__ == "__main__":
    main()
