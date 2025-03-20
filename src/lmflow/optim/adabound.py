#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
from torch.optim.optimizer import Optimizer

class AdaBound(Optimizer):
    r"""Implements AdaBound algorithm.
    
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of
    Learning Rate
    https://arxiv.org/abs/1902.09843
    Note:
        Reference code: https://github.com/Luolc/AdaBound
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsbound: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if final_lr < 0.0:
            raise ValueError(
                "Invalid final learning rate: {}".format(final_lr)
            )
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = [group["lr"] for group in self.param_groups]

    def __setstate__(self, state) -> None:
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    def step(self, closure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        "AdaBound does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                    raise RuntimeError(msg)
                amsbound = group["amsbound"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsbound:
                        # Maintains max of all exp. moving avg. of
                        # sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running
                    # avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"]
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround
                # to apply lr decay
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (
                    1 - 1 / (group["gamma"] * state["step"] + 1)
                )
                upper_bound = final_lr * (
                    1 + 1 / (group["gamma"] * state["step"])
                )
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(
                    exp_avg
                )

                p.data.add_(-step_size)
        return loss