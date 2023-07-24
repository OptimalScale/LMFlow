#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sentencepiece as spm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data/wiki_zh_eval/converted_data.txt', type=str, required=False)
    parser.add_argument('--output_dir', default='./output_models/new_tokenizer', type=str, required=False)
    parser.add_argument('--vocab_size', default=20000, type=int, required=False)
    parser.add_argument('--model_type', default='bpe', type=str, required=False)
    parser.add_argument('--user_defined_symbols', default='0,1,2,3,4,5,6,7,8,9,%', type=str, required=False)
    args = parser.parse_args()    

    dataset_path = args.dataset_path
    output_dir = args.output_dir
    vocab_size = args.vocab_size
    model_type = args.model_type
    user_defined_symbols = args.user_defined_symbols

    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    mkdir(output_dir)

    spm.SentencePieceTrainer.train('--input={} --model_prefix={} --model_type={} --vocab_size={} --user_defined_symbols={} --minloglevel=1'.format(dataset_path,output_dir+'/example',model_type,vocab_size,user_defined_symbols))