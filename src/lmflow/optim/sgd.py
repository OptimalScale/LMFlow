#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Implementation for Stochastic Gradient Descent optimizer."""
from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):  
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):  
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True

            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss
