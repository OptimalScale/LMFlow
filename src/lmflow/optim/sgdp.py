#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch.optim.optimizer import Optimizer


class SGDP(Optimizer):
    r"""Implements SGDP algorithm.

    It has been proposed in `Slowing Down the Weight Norm Increase in
    Momentum-based Optimizers`.
    https://arxiv.org/abs/2006.08217

    Note:
        Reference code: https://github.com/clovaai/AdamP
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
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
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if dampening < 0.0:
            raise ValueError("Invalid dampening value: {}".format(dampening))
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
            momentum=momentum,
            dampening=dampening,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super(SGDP, self).__init__(params, defaults)

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
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                # SGD
                buf = state["momentum"]
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    d_p, wd_ratio = self._projection(
                        p,
                        grad,
                        d_p,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # Weight decay
                if weight_decay != 0:
                    p.data.mul_(
                        1
                        - group["lr"]
                        * group["weight_decay"]
                        * wd_ratio
                        / (1 - momentum)
                    )

                # Step
                p.data.add_(d_p, alpha=-group["lr"])

        return loss