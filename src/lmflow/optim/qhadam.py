#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer

class QHAdam(Optimizer):
    r"""Implements the QHAdam optimization algorithm.

    It has been proposed in `Adaptive methods for Nonconvex Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        nus: immediate discount factors used to estimate the gradient and its
            square (default: (1.0, 1.0))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        decouple_weight_decay: whether to decouple the weight
            decay from the gradient-based optimization step (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.QHAdam(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1810.06801

    Note:
        Reference code: https://github.com/facebookresearch/qhoptim
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        nus = (1.0, 1.0),
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = False,
        eps: float = 1e-8,
    ):
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

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "decouple_weight_decay": decouple_weight_decay,
            "eps": eps,
        }
        super(QHAdam, self).__init__(params, defaults)

    def step(self, closure = None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            decouple_weight_decay = group["decouple_weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError(
                        "QHAdam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        d_p.add_(p.data, alpha=weight_decay)

                d_p_sq = d_p.mul(d_p)

                if len(state) == 0:
                    state["beta1_weight"] = 0.0
                    state["beta2_weight"] = 0.0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                state["beta1_weight"] = 1.0 + beta1 * state["beta1_weight"]
                state["beta2_weight"] = 1.0 + beta2 * state["beta2_weight"]

                beta1_weight = state["beta1_weight"]
                beta2_weight = state["beta2_weight"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(d_p, alpha=1.0 - beta1_adj)
                exp_avg_sq.mul_(beta2_adj).add_(d_p_sq, alpha=1.0 - beta2_adj)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(d_p, alpha=1.0 - nu1)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(d_p_sq, alpha=1.0 - nu2)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                p.data.addcdiv_(avg_grad, avg_grad_rms, value=-lr)

        return loss