#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_dir', default='pinkmanlove/llama-7b-hf', type=str, required=False)
    parser.add_argument('--chinese_sp_model_file', default='./output_models/new_tokenizer/example.model', type=str)
    parser.add_argument('--output_dir', default='./output_models/merged_tokenizer', type=str, required=False)
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir
    chinese_sp_model_file = args.chinese_sp_model_file
    output_dir = args.output_dir
    
    # load
    old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    old_spm = sp_pb2_model.ModelProto()
    old_spm.ParseFromString(old_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
    
    ## Add Chinese tokens to old tokenizer
    old_spm_tokens_set=set(p.piece for p in old_spm.pieces)
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in old_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            old_spm.pieces.append(new_p)

    ## Save
    output_sp_dir = output_dir + '/merged_tokenizer_sp'
    output_hf_dir = output_dir + '/merged_tokenizer_hf' # the path to save tokenizer
    os.makedirs(output_sp_dir,exist_ok=True)
    with open(output_sp_dir+'/merged_tokenizer.model', 'wb') as f:
        f.write(old_spm.SerializeToString())
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir,vocab_file=output_sp_dir+'/merged_tokenizer.model')

    tokenizer.save_pretrained(output_hf_dir)
    logging.info(f"Merged tokenizer has been saved to %s",output_dir)


    # Test
    old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    new_tokenizer = AutoTokenizer.from_pretrained(output_hf_dir)
    logging.info(f"Old tokenizer vocab size: %d",len(old_tokenizer))
    logging.info(f"New tokenizer vocab size: %d",len(new_tokenizer))
    
    text='''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    logging.info(f"Test text:\n %s",text)
    logging.info(f"Tokenized by LLaMA tokenizer:%s",old_tokenizer.tokenize(text))
    logging.info(f"Tokenized by Chinese-LLaMA tokenizer:%s",new_tokenizer.tokenize(text))