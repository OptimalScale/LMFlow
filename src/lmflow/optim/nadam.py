#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math

class NAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=4e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= momentum_decay:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super(NAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_prev'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m_prev, v = state['m_prev'], state['v']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                m = beta1 * m_prev + (1 - beta1) * grad
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                denom = v_hat.sqrt().add_(group['eps'])

                momentum_decay = group['momentum_decay']
                m_prev.mul_(beta1).add_(1 - beta1, grad)
                m_prev_hat = m_prev / bias_correction1

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, m_hat + momentum_decay * m_prev_hat, denom)

        return loss