#!/usr/bin/env python
#coding=utf-8
import argparse
import os
import sys
from vllm import LLM, SamplingParams

def parse_argument(sys_argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.3)
    args = parser.parse_args(sys_argv[1:])
    return args

def main():
    args = parse_argument(sys.argv)
    model = LLM(
        model=args.model,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    while True:
        try:
            prompt = input(">>> ")
            inf_result = model.generate(prompt, sampling_params, use_tqdm=False)
            print(inf_result[0].outputs[0].text)
            print('\n\n')

        except EOFError:
            break

if __name__ == '__main__':
    main()
