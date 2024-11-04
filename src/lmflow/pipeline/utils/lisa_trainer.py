import logging
import time
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled

from lmflow.utils.debug.common import timer


from deepspeed import comm as dist
from deepspeed.runtime.utils import empty_cache, see_memory_usage
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.accelerator import get_accelerator


logger = logging.getLogger(__name__)


LISA_LAYER_NAME_MAPPING = {
    'LlamaForCausalLM': 'model.layers',
    'Qwen2ForCausalLM': 'model.layers',
    'MistralForCausalLM': 'model.layers',
    'MixtralForCausalLM': 'model.layers',
    'GemmaForCausalLM': 'model.layers',
    'GPT2LMHeadModel': 'transformer.h',
}


LISA_BODY_LAYER_PARAM_GROUPS_IDX = [2, 3]


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
        
        self.num_body_layers = len(self._get_all_body_layers())
        self.active_layers_indices = []
        self.active_layers_names = []

        
    def _get_all_body_layers(self) -> List[nn.Module]:
        '''Fetch all the layers of the model excluding the head'''
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        layers = eval('opt_model.' + self.lisa_layer_attr_name)
        return layers
    
    
    def _get_active_layers_names(self) -> List[str]:
        if not hasattr(self, 'active_layers_indices'):
            return []
        
        all_names = []
        layers = self._get_all_body_layers()
        for idx in self.active_layers_indices:
            for name, _ in layers[idx].named_parameters():
                all_names.append(f"{self.lisa_layer_attr_name}.{idx}.{name}")
        
        return all_names
                   

    def _update_active_layer_info(self):
        self.active_layers_indices = np.random.choice(range(self.num_body_layers), self.n_layers, replace=False)
        # self.active_layers_indices.sort()
        # self.active_layers_indices = [3, 4] if self.active_layers_indices == [1, 2] else [1, 2]
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
                    
        if hasattr(self.accelerator, "deepspeed_engine_wrapped"):
            keys_to_remove = [
                statekey for statekey_idx, statekey in enumerate(self.optimizer.state.keys()) 
                if statekey_idx in LISA_BODY_LAYER_PARAM_GROUPS_IDX
            ]
            for key in keys_to_remove:
                del self.optimizer.state[key]    
        else:
            layers = self._get_all_body_layers()
            for active_layer_idx in self.active_layers_indices:
                for name, param in layers[active_layer_idx].named_parameters():
                    print(f"{name=}")
                    del self.optimizer.state[param]
        
        self._switch_active_layers()
        
        # update optimizer pg so that the new layers could be initialized in optimizer.step()
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        
        self.optimizer.param_groups[2]['params'] = [
            p for n, p in opt_model.named_parameters() if (
                n in self.active_layers_names and n in decay_parameters and p.requires_grad)
        ]
        self.optimizer.param_groups[3]['params'] = [
            p for n, p in opt_model.named_parameters() if (
                n in self.active_layers_names and n not in decay_parameters and p.requires_grad)
        ]
        
        if hasattr(self.accelerator, "deepspeed_engine_wrapped"):
            see_memory_usage("[mem usage] before reinit_deepspeed_zero_optimizer_params", force=True)
            self._reinit_deepspeed_zero_optimizer_params(self.accelerator.deepspeed_engine_wrapped.engine.optimizer)
            see_memory_usage("[mem usage] after reinit_deepspeed_zero_optimizer_params", force=True)
                
    
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

        see_memory_usage('[mem usage] after creating optimizer', True)
        return self.optimizer
                
                
    def _inspect_weight(self):
        '''Dev tool'''
        layers = self._get_all_body_layers()
        
        print(layers[self.active_layers_indices[0]].attn.c_attn.weight.shape)
        print(layers[self.active_layers_indices[0]].attn.c_attn.weight)
        
        
    def _reinit_deepspeed_zero_optimizer_params(self, optimizer: DeepSpeedZeroOptimizer):
        num_non_lisa_body_layer_pgs = len(self.optimizer.param_groups) - len(LISA_BODY_LAYER_PARAM_GROUPS_IDX)
        optimizer.bit16_groups = optimizer.bit16_groups[:num_non_lisa_body_layer_pgs]
        optimizer.round_robin_bit16_groups = optimizer.round_robin_bit16_groups[:num_non_lisa_body_layer_pgs]
        optimizer.round_robin_bit16_indices = optimizer.round_robin_bit16_indices[:num_non_lisa_body_layer_pgs]
        optimizer.round_robin_bit16_meta = optimizer.round_robin_bit16_meta[:num_non_lisa_body_layer_pgs]
        optimizer.bit16_groups_flat = optimizer.bit16_groups_flat[:num_non_lisa_body_layer_pgs]
        optimizer.groups_padding = optimizer.groups_padding[:num_non_lisa_body_layer_pgs]
        optimizer.parallel_partitioned_bit16_groups = optimizer.parallel_partitioned_bit16_groups[:num_non_lisa_body_layer_pgs]
        optimizer.single_partition_of_fp32_groups = optimizer.single_partition_of_fp32_groups[:num_non_lisa_body_layer_pgs]
        optimizer.partition_size = optimizer.partition_size[:num_non_lisa_body_layer_pgs]
        optimizer.params_in_partition = optimizer.params_in_partition[:num_non_lisa_body_layer_pgs]
        optimizer.params_not_in_partition = optimizer.params_not_in_partition[:num_non_lisa_body_layer_pgs]
        optimizer.first_offset = optimizer.first_offset[:num_non_lisa_body_layer_pgs]
        optimizer.real_dp_process_group = [optimizer.dp_process_group for i in range(len(self.optimizer.param_groups))]
        
        for i, param_group in enumerate(optimizer.optimizer.param_groups):
            if i in range(num_non_lisa_body_layer_pgs):
                # skip lmhead, ln, etc.
                continue
            
            partition_id = dist.get_rank(group=optimizer.real_dp_process_group[i])

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            trainable_parameters = []
            for param in param_group['params']:
                if param.requires_grad:
                    param.grad_accum = None
                    trainable_parameters.append(param)
            optimizer.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")
            # move all the parameters to cpu to free up GPU space for creating flat buffer

            # Create temp CPU param copies, free accelerator tensors
            orig_group_numel = 0
            for param in optimizer.bit16_groups[i]:
                orig_group_numel += param.numel()
                param.cpu_data = param.data.cpu()
                param.data = torch.empty(1).to(param.device)

            empty_cache()
            see_memory_usage(f"After moving param group {i} to CPU", force=False)

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            if optimizer.round_robin_gradients:
                round_robin_tensors, round_robin_indices = optimizer._round_robin_reorder(
                    optimizer.bit16_groups[i], dist.get_world_size(group=optimizer.real_dp_process_group[i]))
            else:
                round_robin_tensors = optimizer.bit16_groups[i]
                round_robin_indices = list(range(len(optimizer.bit16_groups[i])))

            optimizer.round_robin_bit16_groups.append(round_robin_tensors)
            optimizer.round_robin_bit16_indices.append(round_robin_indices)

            # Create meta tensors list, ordered according to round_robin_tensors
            meta_tensors = []
            for param in round_robin_tensors:
                meta_tensors.append(torch.zeros_like(param.cpu_data, device="meta"))
            optimizer.round_robin_bit16_meta.append(meta_tensors)

            # create flat buffer in CPU
            flattened_buffer = optimizer.flatten_dense_tensors_aligned(
                optimizer.round_robin_bit16_groups[i],
                optimizer.nccl_start_alignment_factor * dist.get_world_size(group=optimizer.real_dp_process_group[i]),
                use_cpu_data=True)

            # free temp CPU params
            for param in optimizer.bit16_groups[i]:
                del param.cpu_data

            # Move CPU flat tensor to the accelerator memory.
            optimizer.bit16_groups_flat.append(flattened_buffer.to(get_accelerator().current_device_name()))
            del flattened_buffer

            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=False)

            # Record padding required for alignment
            if partition_id == dist.get_world_size(group=optimizer.real_dp_process_group[i]) - 1:
                padding = optimizer.bit16_groups_flat[i].numel() - orig_group_numel
            else:
                padding = 0
            optimizer.groups_padding.append(padding)

            if dist.get_rank(group=optimizer.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=False)

            # set model bit16 weight to slices of flattened buffer
            optimizer._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            data_parallel_partitions = optimizer.get_data_parallel_partitions(optimizer.bit16_groups_flat[i], i)
            optimizer.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * optimizer.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            if not optimizer.fp16_master_weights_and_gradients:
                weights_partition = optimizer.parallel_partitioned_bit16_groups[i][partition_id].to(
                    optimizer.device).clone().float().detach()
            else:
                weights_partition = optimizer.parallel_partitioned_bit16_groups[i][partition_id].to(
                    optimizer.device).clone().half().detach()

            if optimizer.cpu_offload:
                weights_partition = get_accelerator().pin_memory(weights_partition)

            optimizer.single_partition_of_fp32_groups.append(weights_partition)

            # Set local optimizer to have flat params of its own partition.
            # After this, the local optimizer will only contain its own partition of params.
            # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
            optimizer.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [optimizer.single_partition_of_fp32_groups[i]]

            partition_size = len(optimizer.bit16_groups_flat[i]) / dist.get_world_size(group=optimizer.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = optimizer.get_partition_info(
                optimizer.round_robin_bit16_groups[i], partition_size, partition_id)

            optimizer.partition_size.append(partition_size)
            optimizer.params_in_partition.append(params_in_partition)
            optimizer.params_not_in_partition.append(params_not_in_partition)
            optimizer.first_offset.append(first_offset)