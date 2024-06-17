from typing import Dict, Any, List, Tuple

from transformers import AutoTokenizer

from lmflow.args import ModelArguments


def check_homogeneity(model_args_list: List[ModelArguments]) -> bool:
    assert all(isinstance(model_args, ModelArguments) for model_args in model_args_list), \
        "model_args_list should be a list of ModelArguments objects."
    assert len(model_args_list) > 1, "model_args_list should have at least two elements."
    
    tokenizer_names = []
    for model_args in model_args_list:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        tokenizer_names.append(tokenizer.__class__.__name__)
    
    return len(set(tokenizer_names)) == 1