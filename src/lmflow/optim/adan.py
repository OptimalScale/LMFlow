#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import List
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

class Adan(Optimizer):
    """Implements a pytorch variant of Adan.

    Adan was proposed in
    Adan : Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models.
    https://arxiv.org/abs/2208.06677

    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.98, 0.92, 0.99),
                 eps=1e-8,
                 weight_decay=0.0,
                 max_grad_norm=0.0,
                 no_prox=False,
                 foreach: bool = True):
        if not 0.0 <= max_grad_norm:
            raise ValueError('Invalid Max grad norm: {}'.format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(
                betas[2]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            no_prox=no_prox,
            foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(
                self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm) + group['eps']

            clip_global_grad_norm = \
                torch.clamp(max_grad_norm / global_grad_norm, max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            pre_grads = []

            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            bias_correction3 = 1.0 - beta3**group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'pre_grad' not in state or group['step'] == 1:
                    # at first step grad wouldn't be clipped
                    # by `clip_global_grad_norm`
                    # this is only to simplify implementation
                    state['pre_grad'] = p.grad

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                pre_grads.append(state['pre_grad'])

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                pre_grads=pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                clip_global_grad_norm=clip_global_grad_norm,
            )
            if group['foreach']:
                copy_grads = _multi_tensor_adan(**kwargs)
            else:
                copy_grads = _single_tensor_adan(**kwargs)

            for p, copy_grad in zip(params_with_grad, copy_grads):
                self.state[p]['pre_grad'] = copy_grad


def _single_tensor_adan(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    copy_grads = []
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        pre_grad = pre_grads[i]

        grad = grad.mul_(clip_global_grad_norm)
        copy_grads.append(grad.clone())

        diff = grad - pre_grad
        update = grad + beta2 * diff

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
        exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)  # n_t

        denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
        update = exp_avg / bias_correction1
        update.add_(beta2 * exp_avg_diff / bias_correction2).div_(denom)

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)
        else:
            param.add_(update, alpha=-lr)
            param.div_(1 + lr * weight_decay)
    return copy_grads


def _multi_tensor_adan(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    if clip_global_grad_norm < 1.0:
        torch._foreach_mul_(grads, clip_global_grad_norm.item())
    copy_grads = [g.clone() for g in grads]

    diff = torch._foreach_sub(grads, pre_grads)
    # NOTE: line below while looking identical gives different result,
    # due to float precision errors.
    # using mul+add produces identical results to single-tensor,
    # using add+alpha doesn't
    # update = torch._foreach_add(grads, torch._foreach_mul(diff, beta2))
    update = torch._foreach_add(grads, diff, alpha=beta2)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, diff, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(
        exp_avg_sqs, update, update, value=1 - beta3)  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    update = torch._foreach_div(exp_avgs, bias_correction1)
    # NOTE: same issue as above.
    # beta2 * diff / bias_correction2 != diff * (beta2 / bias_correction2)  # noqa
    # using faster version by default. uncomment for tests to pass
    # torch._foreach_add_(update, torch._foreach_div(torch._foreach_mul(exp_avg_diffs, beta2), bias_correction2))  # noqa
    torch._foreach_add_(
        update, torch._foreach_mul(exp_avg_diffs, beta2 / bias_correction2))
    torch._foreach_div_(update, denom)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
    else:
        torch._foreach_add_(params, update, alpha=-lr)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    return copy_grads
