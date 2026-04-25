"""
ref: https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
"""

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "get_device_name",
    "get_torch_device",
    "is_accelerate_env",
    "require_cuda_for_gpu_mode",
    "set_cuda_device",
]


def is_accelerate_env():
    """Return True if any environment variable *name* starts with ``ACCELERATE_``."""
    return any(key.startswith("ACCELERATE_") for key in os.environ)


def require_cuda_for_gpu_mode() -> None:
    """Raise if GPU execution was requested but CUDA is not available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available on this machine, but GPU execution was requested. "
            "Install a CUDA-enabled PyTorch build and run on a GPU, or use CPU-compatible "
            "settings where the pipeline supports them."
        )


def set_cuda_device(local_rank: int) -> None:
    """Bind this process to ``local_rank`` on CUDA; raises if CUDA is unavailable."""
    require_cuda_for_gpu_mode()
    torch.cuda.set_device(local_rank)


def get_device_name() -> str:
    """
    Get the device name based on the current machine.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_torch_device() -> Any:
    """Return ``torch.<device_name>`` for the current device name.

    If ``torch`` has no attribute with that name, logs a warning and returns
    ``torch.cuda`` as fallback.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda