#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch.optim.optimizer import Optimizer


class AdamP(Optimizer):
    r"""Implements AdamP algorithm.

    It has been proposed in `Slowing Down the Weight Norm Increase in
    Momentum-based Optimizers`
    https://arxiv.org/abs/2006.08217

    Note:
        Reference code: https://github.com/clovaai/AdamP
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
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
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if delta < 0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if wd_ratio < 0:
            raise ValueError("Invalid wd_ratio value: {}".format(wd_ratio))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super(AdamP, self).__init__(params, defaults)

    @staticmethod
    def _channel_view(x):
        return x.view(x.size(0), -1)

    @staticmethod
    def _layer_view(x):
        return x.view(1, -1)

    @staticmethod
    def _cosine_similarity(x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(
                    expand_size
                ).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(
                    expand_size
                )
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group["betas"]
                nesterov = group["nesterov"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                # Adam
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p,
                        grad,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(
                        1 - group["lr"] * group["weight_decay"] * wd_ratio
                    )

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss