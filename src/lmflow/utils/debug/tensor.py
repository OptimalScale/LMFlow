from typing import List, Tuple

import torch


def tensor_all_zero(tensor: torch.Tensor) -> bool:
    return torch.equal(tensor, torch.zeros_like(tensor))


def find_nonzero_intervals(tensor: torch.Tensor, ignore_distance: int = 1) -> List[Tuple[int, int]]:
    assert tensor.shape == torch.Size([tensor.numel()]), "Input tensor must be 1D"
    assert ignore_distance > 0, "`ignore_distance` must be greater than 0"
    assert ignore_distance < tensor.numel(), "`ignore_distance` must be less than the number of elements in the tensor"
    nonzero_indices = torch.nonzero(tensor, as_tuple=False).squeeze()
    diff = nonzero_indices[1:] - nonzero_indices[:-1]
    non_continuous_points = torch.where(diff > ignore_distance)[0]

    intervals = []
    start = nonzero_indices[0].item()

    for idx in non_continuous_points:
        end = nonzero_indices[idx].item()
        intervals.append((start, end))
        start = nonzero_indices[idx + 1].item()

    # last
    intervals.append((start, nonzero_indices[-1].item()))

    return intervals