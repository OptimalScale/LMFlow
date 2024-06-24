import dataclasses
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""

    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True
)


class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight=None, bias=None, device=None):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        return F.linear(input.to(self.weight.dtype), self.weight, self.bias)


def compress_module(module, target_device):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                CLinear(child.weight, child.bias, target_device),
            )
            compress_module(child, target_device)


def get_compressed_list(module, prefix=""):
    compressed_list = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            full_name = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
            compressed_list.append(full_name)
            compressed_list.extend(
                get_compressed_list(child, full_name)
            )
    return compressed_list


def apply_compressed_weight(module, compressed_state_dict, target_device, prefix=""):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            full_name = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
            setattr(
                module,
                name,
                CLinear(
                    compressed_state_dict[full_name], child.bias, target_device
                ),
            )
            apply_compressed_weight(child, compressed_state_dict, target_device, full_name)


def load_compress_model(model_path, device, torch_dtype):
    # partially load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    base_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(base_pattern)

    config = AutoConfig.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype
    )
    model = AutoModelForCausalLM.from_config(config)
    linear_weights = get_compressed_list(model)

    compressed_state_dict = {}

    for filename in files:
        tmp_state_dict = torch.load(filename)
        for name in tmp_state_dict:
            if name in linear_weights:
                tensor = tmp_state_dict[name].to(device).data.to(torch_dtype)
                compressed_state_dict[name] = compress(
                    tensor, default_compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(device)
            tmp_state_dict[name] = None
            tensor = None
            torch.cuda.empty_cache()

    for name, param in model.named_parameters():
        if name not in linear_weights:
            param.data = compressed_state_dict[name]
    apply_compressed_weight(model, compressed_state_dict, device)

    model.to(device)

    return model, tokenizer


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (
        original_shape[:group_dim]
        + (num_groups, group_size)
        + original_shape[group_dim + 1 :]
    )

    # Pad
    pad_len = group_size - original_shape[group_dim] % group_size
    if pad_len != 0:
        pad_shape = (
            original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1 :]
        )
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim,
        )
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data *= scale

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data += mn

    # Unpad
    pad_len = group_size - original_shape[group_dim] % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim]
            + (original_shape[group_dim] + pad_len,)
            + original_shape[group_dim + 1 :]
        )
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)
