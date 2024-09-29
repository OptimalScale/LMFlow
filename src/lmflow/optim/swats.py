#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer

class SWATS(Optimizer):
    r"""Implements SWATS Optimizer Algorithm.
    It has been proposed in `Improving Generalization Performance by
    Switching from Adam to SGD`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-3)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`
            (default: False)
        nesterov: enables Nesterov momentum (default: False)


    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.SWATS(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/pdf/1712.07628.pdf

    Note:
        Reference code: https://github.com/Mrpatekful/swats
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0,
        amsgrad: bool = False,
        nesterov: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
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
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            phase="ADAM",
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            nesterov=nesterov,
        )

        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("nesterov", False)

    def step(self, closure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for w in group["params"]:
                if w.grad is None:
                    continue
                grad = w.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                amsgrad = group["amsgrad"]

                state = self.state[w]

                # state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        w.data, memory_format=torch.preserve_format
                    )
                    # exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        w.data, memory_format=torch.preserve_format
                    )
                    # moving average for the non-orthogonal projection scaling
                    state["exp_avg2"] = w.new(1).fill_(0)
                    if amsgrad:
                        # maintains max of all exp. moving avg.
                        # of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            w.data, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg2, exp_avg_sq = (
                    state["exp_avg"],
                    state["exp_avg2"],
                    state["exp_avg_sq"],
                )

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(w.data, alpha=group["weight_decay"])

                # if its SGD phase, take an SGD update and continue
                if group["phase"] == "SGD":
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(
                            grad
                        ).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(beta1).add_(grad)
                        grad = buf

                    grad.mul_(1 - beta1)
                    if group["nesterov"]:
                        grad.add_(buf, alpha=beta1)

                    w.data.add_(grad, alpha=-group["lr"])
                    continue

                # decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # maintains the maximum of all 2nd
                    # moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"] * (bias_correction2**0.5) / bias_correction1
                )

                p = -step_size * (exp_avg / denom)
                w.data.add_(p)

                p_view = p.view(-1)
                pg = p_view.dot(grad.view(-1))

                if pg != 0:
                    # the non-orthognal scaling estimate
                    scaling = p_view.dot(p_view) / -pg
                    exp_avg2.mul_(beta2).add_(scaling, alpha=1 - beta2)

                    # bias corrected exponential average
                    corrected_exp_avg = exp_avg2 / bias_correction2

                    # checking criteria of switching to SGD training
                    if (
                        state["step"] > 1
                        and corrected_exp_avg.allclose(scaling, rtol=1e-6)
                        and corrected_exp_avg > 0
                    ):
                        group["phase"] = "SGD"
                        group["lr"] = corrected_exp_avg.item()
        return loss

