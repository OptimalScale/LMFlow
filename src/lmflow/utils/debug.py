import os
import re
import textwrap
from typing import Dict, Any, List, Tuple, Union, Optional

import pandas as pd
import tabulate
import torch
import torch.nn as nn
from accelerate.utils import DistributedType
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support, _device_has_foreach_support
from transformers import PreTrainedModel


def get_distributed_type():
    distributed_type = DistributedType.DEEPSPEED if "ACCELERATE_USE_DEEPSPEED" in os.environ else DistributedType.NO
    return distributed_type


def print_tabulate_with_header(tabulate_df, header: Optional[str] = None):
    if header:
        df_len = len(tabulate_df.split('\n')[0])
        print('\n')
        print('+' + '-'*(df_len-2) + '+')
        wrap_header = textwrap.wrap(header, df_len-4)
        for header in wrap_header:
            print("|" + header.center(df_len-2, ' ') + "|")
            
    print(tabulate_df)


def get_decay_parameter_names(model: Union[PreTrainedModel, nn.Module]) -> List[str]:
    """
    From transformers.trainer
    
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names
    
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def get_parameter_names_in_param_groups(
    model: Union[PreTrainedModel, nn.Module], 
    ignore_requires_grad: bool = True
) -> List[Dict[str, str]]:
    decay_parameters = get_decay_parameter_names(model)
    
    if ignore_requires_grad:
        parameter_names = [
            {
                "parameter_names": [
                    n for n, p in model.named_parameters() if (n in decay_parameters)
                ],
            },
            {
                "parameter_names": [
                    n for n, p in model.named_parameters() if (n not in decay_parameters)
                ],
            },
        ]
    else:
        parameter_names = [
            {
                "parameter_names": [
                    n for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "parameter_names": [
                    n for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
            },
        ]
    
    return parameter_names


def get_parameter_names_require_grads(
    model: Union[PreTrainedModel, nn.Module],
) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def guess_grad_norms_from_pg(
    parameter_names: List[Dict[str, str]],
    all_norms: List[torch.Tensor],
    show_zero_grads: bool = False,
    separate_by_layer: bool = False,
):
    all_grad_norms = {
        "name": [],
        "layer": [],
        "grad_norm": [],
    }
    has_guess = False
    pg_note = None
    
    for pg_idx, pg_names in enumerate(parameter_names):
        if len(pg_names["parameter_names"]) == len(all_norms):
            all_grad_norms["name"] = pg_names["parameter_names"]
            all_grad_norms["grad_norm"] = [norm_tensor.item() for norm_tensor in all_norms]
            if not has_guess:
                has_guess = True
                pg_note = 'Parameter group with weight decay' if pg_idx == 0 else 'Parameter group without weight decay'
            else:
                print("Failed to guess grad norms from parameter groups according to group length.")
                return
            
    if not has_guess:
        return
    
    layer_pattern = re.compile(r'transformer\.h\.(\d+)\.')
    for name in all_grad_norms["name"]:
        layer_match = layer_pattern.search(name)
        if layer_match:
            all_grad_norms["layer"].append(int(layer_match.group(1)))
        else:
            all_grad_norms["layer"].append('other')
            
    df = pd.DataFrame(all_grad_norms)
    if not show_zero_grads:
        df = df[df["grad_norm"] > 0.0]
        
    if not separate_by_layer:
        table_to_print = tabulate.tabulate(df, headers='keys', tablefmt='psql', showindex=False)
        print_tabulate_with_header(table_to_print, pg_note)
    else:
        for layer_idx in df["layer"].unique():
            table_to_print = tabulate.tabulate(
                df[df["layer"] == layer_idx], headers='keys', tablefmt='psql', showindex=False
            )
            print_tabulate_with_header(table_to_print, f"Layer {layer_idx}, {pg_note}")
            
            
def guess_grad_norms_from_hf_trainer(
    parameter_names: List[str],
    all_norms: List[torch.Tensor],
    separate_by_layer: bool = False,
    note: Optional[str] = None
):
    all_grad_norms = {
        "name": parameter_names,
        "layer": [],
        "grad_norm": [norm_tensor.item() for norm_tensor in all_norms],
    }
    
    layer_pattern = re.compile(r'transformer\.h\.(\d+)\.')
    for name in all_grad_norms["name"]:
        layer_match = layer_pattern.search(name)
        if layer_match:
            all_grad_norms["layer"].append(int(layer_match.group(1)))
        else:
            all_grad_norms["layer"].append('other')
    
    df = pd.DataFrame(all_grad_norms)
        
    if not separate_by_layer:
        table_to_print = tabulate.tabulate(df, headers='keys', tablefmt='psql', showindex=False)
        print_tabulate_with_header(table_to_print, note)
    else:
        for layer_idx in df["layer"].unique():
            table_to_print = tabulate.tabulate(
                df[df["layer"] == layer_idx], headers='keys', tablefmt='psql', showindex=False
            )
            print_tabulate_with_header(table_to_print, f"Layer {layer_idx}, {note}")