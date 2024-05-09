import torch
import torch.nn as nn
import torch.nn.functional as F
import math,random
import numpy as np
import accelerate


class LISADiffusion:
    def __init__(self, model, rate=None):
        self.model = model
        self.rate = rate
        self.initialize()

    def freeze_all_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def random_activate_layers(self, model, p):
        activate_number = int((len(list(model.parameters()))-2) * p)
        index = np.random.choice(range(0,len(list(model.parameters()))-1,1), activate_number, replace=False)
        count = 0
        for param in model.parameters():
            if count == 0 or count == len(list(model.parameters()))-1:
                param.requires_grad = True
            elif count in index:
                param.requires_grad = True            
            count += 1

    def lisa(self, model, p=0.25):
        self.freeze_all_layers(model)
        self.random_activate_layers(model, p)

    def lisa_recall(self):
        param_number = len(list(self.model.parameters()))
        lisa_p = 8 / param_number if self.rate is None else self.rate
        self.lisa(model=self.model,p=lisa_p)

    def initialize(self):
        self.optimizer_dict = dict()
        self.scheduler_dict = dict()
        self.lisa_recall()

    def register(self, optimizer_class=None, get_scheduler=None, accelerator=None, 
                 optim_kwargs={}, sched_kwargs={}):
        for p in self.model.parameters():
            if p.requires_grad:
                self.optimizer_dict[p] = optimizer_class([{"params":p}], **optim_kwargs)
                if accelerator is not None:
                    self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])

        for p in self.model.parameters():
            if p.requires_grad:
                self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                if accelerator is not None:
                    self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])
    
    def insert_hook(self, optimizer_class=None, get_scheduler=None, accelerator=None, 
                 optim_kwargs={}, sched_kwargs={}):
        def optimizer_hook(p):
            if p.grad is None:
                del self.scheduler_dict[p]
                del self.optimizer_dict[p]
                return
            else:
                if p not in self.optimizer_dict:
                    self.optimizer_dict[p] = optimizer_class([{"params":p}], **optim_kwargs)
                    if accelerator is not None:
                        self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])    
                if p not in self.scheduler_dict:
                    self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                    if accelerator is not None:
                        self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])

            if accelerator is not None and accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(p, 10.0)
            
            self.optimizer_dict[p].step()
            self.optimizer_dict[p].zero_grad(set_to_none=True)
            self.scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
