#!/usr/bin/env python
# coding=utf-8
"""Dummy Optimizer.
"""
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

class Dummy(Optimizer):
    """
    An dummy optimizer that does nothing.

    Parameters:
        params (:obj:`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 0):
            The learning rate to use.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 0.,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure: Callable=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Dummy does not support sparse gradients yet")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg2"] = torch.zeros_like(p)

                # v := exp_avg
                # m := double_exp_avg
                v, m = state["exp_avg"], state["exp_avg2"]
                beta1, beta2 = group["betas"]
                step_size = group["lr"]

                state["step"] += 1

                p.add_(m, alpha=-0.0)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        return loss
