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


def make_shell_args_from_dataclass(
    dataclass_objects: List, 
    format: str="subprocess",
    skip_default: bool=True,
) -> Union[str, List[str]]:
    """Return a string or a list of strings that can be used as shell arguments.

    Parameters
    ----------
    dataclass_objects : List
        A list of dataclass objects.
    format : str, optional
        Return format, can be "shell" or "subprocess", by default "subprocess".
    skip_default : bool, optional
        Whether to skip attributes with default values, by default True. 

    Returns
    -------
    Union[str, List[str]]
    """
    assert isinstance(dataclass_objects, list), "dataclass_objects should be a list of dataclass objects."
    all_args = {}
    for dataclass_object in dataclass_objects:
        for k, v in dataclass_object.__dict__.items():
            if not v:
                continue
            if skip_default:
                if dataclass_object.__dataclass_fields__[k].default == v:
                    continue
            if k not in all_args:
                all_args[k] = v
            elif k in all_args:
                if all_args[k] == v:
                    continue
                else:
                    logger.warning(f"Found different values for the same key: {k}, using value: {v} instead.")
                    all_args[k] = v
    
    if format == "shell":
        final_res = " ".join([f"--{k} {v}" for k, v in all_args.items()])
    elif format == "subprocess":
        final_res = []
        for k, v in all_args.items():
            final_res.extend([f"--{k}", str(v)])
    else:
        raise ValueError(f"Unknown format: {format}")
        
    return final_res
