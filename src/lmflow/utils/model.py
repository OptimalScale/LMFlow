#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
from typing import Dict, Any, List, Tuple, Union

from transformers import AutoTokenizer

from lmflow.args import ModelArguments


logger = logging.getLogger(__name__)


def check_homogeneity(model_args_list: List[ModelArguments]) -> bool:
    assert all(isinstance(model_args, ModelArguments) for model_args in model_args_list), \
        "model_args_list should be a list of ModelArguments objects."
    assert len(model_args_list) > 1, "model_args_list should have at least two elements."
    
    tokenizer_names = []
    for model_args in model_args_list:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        tokenizer_names.append(tokenizer.__class__.__name__)
    
    return len(set(tokenizer_names)) == 1