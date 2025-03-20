#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Implementation for the RMSprop algorithm."""
from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

class RMSprop(Optimizer):  
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered=False,
        capturable=False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):  # noqa: D107
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            capturable=capturable,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=torch.get_default_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=torch.get_default_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        square_avgs,
        momentum_buffer_list,
        grad_avgs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)

            if p.grad.is_sparse:
                raise RuntimeError("RMSprop does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=torch.get_default_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.zeros((), dtype=torch.get_default_dtype())
                )
                state["square_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group["momentum"] > 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if group["centered"]:
                    state["grad_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            square_avgs.append(state["square_avg"])
            state_steps.append(state["step"])

            if group["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])
            if group["centered"]:
                grad_avgs.append(state["grad_avg"])

        return has_complex

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            square_avgs: List[Tensor] = []
            grad_avgs: List[Tensor] = []
            momentum_buffer_list: List[Tensor] = []
            state_steps: List[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                square_avgs,
                momentum_buffer_list,
                grad_avgs,
                state_steps,
            )

        return loss