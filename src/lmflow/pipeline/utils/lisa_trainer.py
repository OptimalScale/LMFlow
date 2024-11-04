import logging
from typing import Union, List

import torch.nn as nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled

from lmflow.utils.debug.common import timer


logger = logging.getLogger(__name__)


LISA_LAYER_NAME_MAPPING = {
    'LlamaForCausalLM': 'model.layers',
    'Qwen2ForCausalLM': 'model.layers',
    'MistralForCausalLM': 'model.layers',
    'MixtralForCausalLM': 'model.layers',
    'GemmaForCausalLM': 'model.layers',
    'GPT2LMHeadModel': 'transformer.h',
}


LISA_PARAM_GROUPS_KEEP_STATE_IDX = [0, 1]


class LISATrainer(Trainer):
    def __init__(
        self, 
        n_layers: int, 
        interval_steps: int, 
        lisa_layer_attr_name: str = None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        setattr(self.args, '_trainer', self) # make trainer callbacks accessible to the attributes in trainer
        
        # lisa specific attributes
        self.n_layers = n_layers
        self.interval_steps = interval_steps
        
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        model_class_name = opt_model.__class__.__name__
        if model_class_name in LISA_LAYER_NAME_MAPPING:
            self.lisa_layer_attr_name = LISA_LAYER_NAME_MAPPING[model_class_name]
        else:
            assert lisa_layer_attr_name is not None, "Please provide the attribute name for the model layers."
            self.lisa_layer_attr_name = lisa_layer_attr_name
        
        self.num_body_layers = len(self._get_all_body_layers())  # Dynamically execute to get the number of layers
        self.active_layers_indices = []
        self.active_layers_names = []

        
    def _get_all_body_layers(self) -> List[nn.Module]:
        '''Fetch all the layers of the model excluding the head'''
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        layers = eval('opt_model.' + self.lisa_layer_attr_name)
        return layers
    
    
    def _get_active_layers_names(self) -> List[str]:
        if not self.active_layers_indices:
            return []
        
        all_names = []
        layers = self._get_all_body_layers()        
        for idx in self.active_layers_indices:
            for name, _ in layers[idx].named_parameters():
                all_names.append(f"{self.lisa_layer_attr_name}.{idx}.{name}")
        
        return all_names
                   

    def _update_active_layer_info(self):
        # self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
        # self.active_layers_indices.sort()
        self.active_layers_indices = [3, 4] if self.active_layers_indices == [1, 2] else [1, 2]
        self.active_layers_names = self._get_active_layers_names()
        print(f"Layers for the next steps: {self.active_layers_indices}: {self.active_layers_names}", flush=True)
        
        
    def _switch_active_layers(self):
        '''
        Switch the active layers for the next interval. Objects that will be updated after calling:
        1. self.active_layers_indices
        2. self.active_layers_names
        3. requires_grad of the parameters
        '''
        # Disable gradients for all layers
        layers = self._get_all_body_layers()
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
                
        # Randomly select n_layers to activate
        self._update_active_layer_info() # update active name and idx
        
        # Enable gradients only for the selected layers
        layers = self._get_all_body_layers()  # Re-fetch layer references
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True
                
                
    def maybe_switch_active_layers(self):
        if (
            self.state.global_step == 0 # skip since already initialized in `create_optimizer`
            or 
            self.state.global_step % self.interval_steps != 0 
        ):
            return
        
        print('maybe_switch_active_layers')
        # # clear optimizer.state
        # print('Clearing optimizer state...')
        # print(f"{self.optimizer.param_groups=}")
        # layers = self._get_all_body_layers()
        # for active_layer_idx in self.active_layers_indices:
        #     for name, param in layers[active_layer_idx].named_parameters():
        #         print(f"{name=}")
        #         del self.optimizer.state[param]
        # print(f"{len(self.optimizer.state.keys())=}")
        
        # self._switch_active_layers()
        
        # # update optimizer pg so that the new layers could be initialized in optimizer.step()
        # opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        # decay_parameters = self.get_decay_parameter_names(opt_model)
        
        # self.optimizer.param_groups[2]['params'] = [
        #     p for n, p in opt_model.named_parameters() if (
        #         n in self.active_layers_names and n in decay_parameters and p.requires_grad)
        # ]
        # self.optimizer.param_groups[3]['params'] = [
        #     p for n, p in opt_model.named_parameters() if (
        #         n in self.active_layers_names and n not in decay_parameters and p.requires_grad)
        # ]
        
        # print(f"{self.optimizer.param_groups=}")
        
        print('reset')
        print(f"keys before reset {self.optimizer.state.keys()=}")
        optimizer_cache = self.optimizer
        self.optimizer = None
        if hasattr(self.accelerator, "deepspeed_engine_wrapped"):
            if self.accelerator.deepspeed_engine_wrapped is not None:
                self.accelerator.deepspeed_engine_wrapped.engine.destroy()
            self.accelerator.deepspeed_engine_wrapped = None
        self.create_optimizer()
        _, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        print(f"keys after perpare {self.optimizer.state.keys()=}")
        for gindex in LISA_PARAM_GROUPS_KEEP_STATE_IDX:
            for param in optimizer_cache.param_groups[gindex]['params']:
                self.optimizer.state[param] = optimizer_cache.state[param]
                
        print(f"keys after reset {self.optimizer.state.keys()=}")
                
    
    @timer
    def create_optimizer(self):
        """
        Setup the optimizer. Adopted from transformers.Trainer.create_optimizer.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            self._switch_active_layers() # init along with the optimizer
            
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    # this should always be lmhead:
                    # `requires_grad` and `not in active_layers_names` rules out all body layers
                    # `in decay_parameters` rules out ln
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            n not in self.active_layers_names and n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    # this should always be ln (outside of body layers)
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            n not in self.active_layers_names and n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    # selected body layers with decay 
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            n in self.active_layers_names and n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    # selected body layers without decay
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            n in self.active_layers_names and n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_grouped_parameter_names = [
                {
                    "param_names": [
                        n for n, p in opt_model.named_parameters() if (n not in self.active_layers_names and n in decay_parameters and p.requires_grad)
                    ],
                },
                {
                    "param_names": [
                        n for n, p in opt_model.named_parameters() if (n not in self.active_layers_names and n not in decay_parameters and p.requires_grad)
                    ],
                },
                {
                    "param_names": [
                        n for n, p in opt_model.named_parameters() if (n in self.active_layers_names and n in decay_parameters and p.requires_grad)
                    ],
                }, # lisa active layers with decay
                {
                    "param_names": [
                        n for n, p in opt_model.named_parameters() if (n in self.active_layers_names and n not in decay_parameters and p.requires_grad)
                    ],
                }, # lisa active layers without decay
            ]
            print(optimizer_grouped_parameter_names)            

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        print(self.optimizer)
        return self.optimizer
                
                
    def _inspect_weight(self):
        '''Dev tool'''
        layers = self._get_all_body_layers()
        
        print(layers[self.active_layers_indices[0]].attn.c_attn.weight.shape)
        print(layers[self.active_layers_indices[0]].attn.c_attn.weight)
        print(self.model.transformer.wte.weight)