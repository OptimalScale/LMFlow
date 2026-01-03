"""
ref: https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
"""

import os
import logging

import torch


logger = logging.getLogger(__name__)
is_cuda_available = torch.cuda.is_available()


def is_accelerate_env():
    for key, _ in os.environ.items():
        if key.startswith("ACCELERATE_"):
            return True
    return False


def get_device_name() -> str:
    """
    Get the device name based on the current machine.
    """
    if is_cuda_available:
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda