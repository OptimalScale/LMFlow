#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import textwrap
from typing import Dict, Any, List, Tuple, Union, Optional

import pandas as pd
import tabulate
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


def print_tabulate_with_header(tabulate_df, header: Optional[str] = None):
    if header:
        df_len = len(tabulate_df.split('\n')[0])
        print('+' + '-'*(df_len-2) + '+')
        wrap_header = textwrap.wrap(header, df_len-4)
        for header in wrap_header:
            print("|" + header.center(df_len-2, ' ') + "|")
            
    print(tabulate_df)


def inspect_layer(layers, layer_idx: int, note: Optional[str] = None):
    layer_info = {
        "name": [],
        "size": [],
        "requires_grad": []
    }
    
    for n, p in layers[layer_idx].named_parameters():
        layer_info["name"].append(n)
        layer_info["size"].append(p.size())
        layer_info["requires_grad"].append(p.requires_grad)
    
    df = pd.DataFrame(layer_info)
    table_to_print = tabulate.tabulate(df, headers='keys', tablefmt='psql')
    
    print_tabulate_with_header(table_to_print, note)


def inspect_layers(
    layers, 
    layer_idxs: Union[int, List[int]], 
    notes: Optional[Union[str, List[str]]] = None
):
    if isinstance(layer_idxs, int):
        layer_idxs = [layer_idxs]
    if notes:
        if isinstance(notes, str):
            notes = [notes]
    assert len(layer_idxs) == len(notes) if notes else True
    
    for layer_idx in layer_idxs:
        inspect_layer(layers, layer_idx, notes[layer_idx] if notes else None)


def check_layerwise_requires_grad(
    layers, 
    layer_idx: Union[str, int, List[int]] = 'all', 
    show_details: bool = False,
    note: Optional[Union[str, List[str]]] = None
):
    if layer_idx == 'all':
        layer_idx = list(range(len(layers)))
    elif isinstance(layer_idx, int):
        layer_idx = [layer_idx]
        
    all_states = {
        "layer_idx": layer_idx,
        "requires_grad": [],
    }
    
    for idx in layer_idx:
        layer_states = {
            "names": [],
            "requires_grad": [],
            "requires_grad_meta": False
        }
        for n, p in layers[idx].named_parameters():
            layer_states["names"].append(n)
            layer_states["requires_grad"].append(p.requires_grad)
        if all(layer_states["requires_grad"]):
            layer_states["requires_grad_meta"] = True
        else:
            if show_details:
                inspect_layer(layers, idx, f"Layer {idx}")
        all_states["requires_grad"].append(layer_states['requires_grad_meta'])
    
    df = pd.DataFrame(all_states)
    table_to_print = tabulate.tabulate(df, headers='keys', tablefmt='psql', showindex=False)
    
    print_tabulate_with_header(table_to_print, note)