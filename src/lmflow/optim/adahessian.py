#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import List, Optional
import torch
from torch.optim.optimizer import Optimizer

class Adahessian(Optimizer):
    r"""Implements Adahessian Algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer
    for Machine Learning`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 0.5)
        seed (int, optional): Random number generator seed (default: None)

        Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Adahessian(model.parameters(), lr = 1.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward(create_graph=True)
        >>> optimizer.step()

        __ https://arxiv.org/abs/2006.00719

        Note:
            Reference code: https://github.com/amirgholami/adahessian
    """

    def __init__(
        self,
        params,
        lr: float = 0.15,
        betas = (0.9, 0.999),
        eps: float = 1e-4,
        weight_decay: float = 0,
        hessian_power: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(
                "Invalid Hessian power value: {}".format(hessian_power)
            )
        if seed is not None:
            torch.manual_seed(seed)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads) -> List[torch.Tensor]:
        """Get an estimate of Hessian Trace.
        This is done by computing the Hessian vector product with a random
        vector v at the current gradient point, to estimate Hessian trace by
        computing the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                msg = (
                    "Gradient tensor {:} does not have grad_fn. When "
                    "calling loss.backward(), make sure the option "
                    "create_graph is set to True."
                )
                raise RuntimeError(msg.format(i))

        v = [
            2
            * torch.randint_like(
                p, high=2, memory_format=torch.preserve_format
            )
            - 1
            for p in params
        ]

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        hvs = torch.autograd.grad(
            grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces
                # the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        return hutchinson_trace

    def step(self, closure = None):
        """Perform a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal

        hut_traces = self.get_trace(params, grads)

        for p, group, grad, hut_trace in zip(
            params, groups, grads, hut_traces
        ):
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = (
                state["exp_avg"],
                state["exp_hessian_diag_sq"],
            )

            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(
                hut_trace, hut_trace, value=1 - beta2
            )

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # make the square root, and the Hessian power
            k = group["hessian_power"]
            denom = (
                (exp_hessian_diag_sq.sqrt() ** k)
                / math.sqrt(bias_correction2) ** k
            ).add_(group["eps"])

            # make update
            p.data = p.data - group["lr"] * (
                exp_avg / bias_correction1 / denom
                + group["weight_decay"] * p.data
            )

        return loss