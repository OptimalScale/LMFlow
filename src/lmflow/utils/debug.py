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
        print('+' + '-'*(df_len-2) + '+')
        wrap_header = textwrap.wrap(header, df_len-4)
        for header in wrap_header:
            print("|" + header.center(df_len-2, ' ') + "|")
            
    print(tabulate_df)


def inspect_layer(layers, layer_idx: int, note: Optional[str] = None):
    layer_info = {
        "name": [],
        "size": [],
        "requires_grad": [],
        "grad_norm": [],
    }
    
    for n, p in layers[layer_idx].named_parameters():
        layer_info["name"].append(n)
        layer_info["size"].append(p.size())
        layer_info["requires_grad"].append(p.requires_grad)
        
    layer_info["grad_norm"] = [
        norm_tensor.item() 
        for norm_tensor in clip_grad_norm_(parameters=layers[layer_idx].parameters(), 
                                           max_norm=1.0, 
                                           return_norm_by_layer=True)[1]
    ]
    
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


def check_layerwise_grad(
    layers, 
    layer_idx: Union[str, int, List[int]] = 'all', 
    show_details: Optional[str] = 'has_grads',
    note: Optional[Union[str, List[str]]] = None,
):
    if layer_idx == 'all':
        layer_idx = list(range(len(layers)))
    elif isinstance(layer_idx, int):
        layer_idx = [layer_idx]
        
    distributed_type = get_distributed_type()
        
    all_states = {
        "layer_idx": layer_idx,
        "requires_grad": [],
        "grad_norm": []
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
            
        if show_details == 'all':
            inspect_layer(layers, idx, f"Layer {idx} detail")
        elif show_details == 'has_grads':
            if any(layer_states["requires_grad"]):
                inspect_layer(layers, idx, f"Layer {idx} detail")
            
        all_states["requires_grad"].append(layer_states['requires_grad_meta'])
        all_states["grad_norm"].append(clip_grad_norm_(layers[idx].parameters(), 1.0, distributed_type=distributed_type).item())
    
    df = pd.DataFrame(all_states)
    table_to_print = tabulate.tabulate(df, headers='keys', tablefmt='psql', showindex=False)
    
    print_tabulate_with_header(table_to_print, f"{note}, {distributed_type=}")
    
    
def clip_grad_norm_(
        parameters, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None,
        distributed_type: DistributedType = DistributedType.NO,
        return_norm_by_layer: bool = False
    ) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if distributed_type == DistributedType.DEEPSPEED:
        # from deepspeed.utils import safe_get_full_grad
        # grads = [safe_get_full_grad(p) for p in parameters]
        return torch.tensor(0.)
    else:
        grads = [p.grad for p in parameters if p.grad is not None]
    # print(f'torch grads {grads=}')
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: List[Tensor] = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    # print(f'torch norms {norms=}')
    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    # print(f'torch total_norm {total_norm=}')

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    # print(f'torch total_norm at end {total_norm=}')
    return (total_norm, norms) if return_norm_by_layer else total_norm


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
    for pg_names in parameter_names:
        if len(pg_names["parameter_names"]) == len(all_norms):
            all_grad_norms["name"] = pg_names["parameter_names"]
            all_grad_norms["grad_norm"] = [norm_tensor.item() for norm_tensor in all_norms]
            
    if not all_grad_norms["name"]:
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
        print_tabulate_with_header(table_to_print)
    else:
        for layer_idx in df["layer"].unique():
            table_to_print = tabulate.tabulate(
                df[df["layer"] == layer_idx], headers='keys', tablefmt='psql', showindex=False
            )
            print_tabulate_with_header(table_to_print, f"Layer {layer_idx}")