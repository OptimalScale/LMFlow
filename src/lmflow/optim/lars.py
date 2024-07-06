#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    r"""Extends SGD in PyTorch with LARS scaling from the paper
    `Large batch training of Convolutional Networks`__.
    .. note::
        The application of momentum in the SGD part is modified according to
        the PyTorch standards. LARS scaling fits into the equation in the
        following fashion.

        .. math::
            \begin{aligned}
                g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
                v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \\end{aligned}

        where :math:`p`, :math:`g`, :math:`v`, :math:`\\mu` and :math:`\beta`
        denote the parameters, gradient, velocity, momentum, and weight decay
        respectively.  The :math:`lars_lr` is defined by Eq. 6 in the paper.
        The Nesterov version is analogously modified.

    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded
        from layer-wise LR scaling. This is to ensure consistency with papers
        like SimCLR and BYOL.


    __ https://arxiv.org/pdf/1708.03888.pdf

    Note:
        Reference code: https://github.com/PyTorchLightning/lightning-bolts/
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        trust_coefficient: float = 0.01,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if dampening < 0.0:
            raise ValueError("Invalid dampening value: {}".format(dampening))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if trust_coefficient < 0.0:
            raise ValueError(
                "Invalid trust_coefficient value: {}".format(trust_coefficient)
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )

        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["trust_coefficient"]

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss