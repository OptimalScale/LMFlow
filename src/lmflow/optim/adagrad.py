#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(AdaGrad, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                state = self.state[p]

                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum = state['sum']
                sum.addcmul_(1, grad, grad)
                std = sum.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], grad, std)

        return loss