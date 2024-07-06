#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer

class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.95, eps=1e-6):
        defaults = dict(lr=lr, rho=rho, eps=eps)
        super(Adadelta, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['acc_delta'] = torch.zeros_like(p.data)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']

                state['step'] += 1

                square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)

                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)

                p.data.add_(-delta)

                acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)

        return loss