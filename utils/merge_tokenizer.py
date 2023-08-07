#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoTokenizer,LlamaTokenizer

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_dir', default='openlm-research/open_llama_3b', type=str, required=False)
    parser.add_argument('--chinese_sp_model_file', default='./output_models/new_tokenizer/example.model', type=str)
    parser.add_argument('--output_dir', default='./output_models/merged_tokenizer', type=str, required=False)
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir
    chinese_sp_model_file = args.chinese_sp_model_file
    output_dir = args.output_dir
    
    # load
    try:
        old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    except RecursionError:
        old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                    unk_token="<unk>",
                                                    bos_token="<s>",
                                                    eos_token="</s>",
                                                    use_fast=False)
        
    if not isinstance(old_tokenizer,LlamaTokenizer):
        raise ValueError("The tokenizer is not a LlamaTokenizer, we only support LlamaTokenizer for now.")

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
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_dir,
            vocab_file=output_sp_dir+'/merged_tokenizer.model',
            use_fast=False
        )
    except RecursionError:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir,
                                                unk_token="<unk>",
                                                bos_token="<s>",
                                                eos_token="</s>",
                                                vocab_file=output_sp_dir+'/merged_tokenizer.model',
                                                use_fast=False)

    tokenizer.save_pretrained(output_hf_dir)
    logging.info(f"Merged tokenizer has been saved to %s",output_dir)


    # Test
    new_tokenizer = tokenizer
    logging.info(f"Old tokenizer vocab size: %d",len(old_tokenizer))
    logging.info(f"New tokenizer vocab size: %d",len(new_tokenizer))
    
    text='''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    logging.info(f"Test text:\n %s",text)
    logging.info(f"Tokenized by original tokenizer:%s",old_tokenizer.tokenize(text))
    logging.info(f"Tokenized by merged tokenizer:%s",new_tokenizer.tokenize(text))