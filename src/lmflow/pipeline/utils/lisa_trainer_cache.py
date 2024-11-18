import gc
import logging
import time
from collections import defaultdict
from typing import Union, List, DefaultDict, Any

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
torch.cuda.memory._record_memory_history(max_entries=100000)


LISA_LAYER_NAME_MAPPING = {
    'LlamaForCausalLM': 'model.layers',
    'Qwen2ForCausalLM': 'model.layers',
    'MistralForCausalLM': 'model.layers',
    'MixtralForCausalLM': 'model.layers',
    'GemmaForCausalLM': 'model.layers',
    'GPT2LMHeadModel': 'transformer.h',
}


LISA_BODY_LAYER_PARAM_GROUPS_IDX = [0, 1]
NON_LISA_LAYER_PARAM_GROUPS_IDX = [2, 3]


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
        self.histroy_layers_indices = []
        self.active_layers_names = []
        self._optimizer_param_group_initialized = False

        
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
        # self.active_layers_indices = [3, 4] if self.active_layers_indices == [1, 2] else [1, 2]
        # self.active_layers_indices = [1, 2]
        self.active_layers_indices = np.random.choice(range(self.num_body_layers), self.n_layers, replace=False)
        self.histroy_layers_indices.append(list(self.active_layers_indices))
        # self.active_layers_indices.sort()
        self.active_layers_names = self._get_active_layers_names()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
        print(f"History of layers: {self.histroy_layers_indices[:-1]}", flush=True)
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
        # if self.state.global_step == 0:
        #     torch.cuda.memory._dump_snapshot(f'gs_{self.state.global_step}.pickle')
            
        if (
            self.state.global_step == 0 # skip since already initialized in `create_optimizer`
            or 
            self.state.global_step % self.interval_steps != 0 
        ):
            return
        
        # cache param groups that don't need to swtich (lmhead, ln)
        non_lisa_param_groups = [self.optimizer.param_groups[i] for i in NON_LISA_LAYER_PARAM_GROUPS_IDX]
        
        # cache states of non-lisa layers
        non_lisa_states: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        for pg in non_lisa_param_groups:
            for param in pg['params']:
                non_lisa_states[param] = self.optimizer.state[param]            
        
        # clear optimizer to clear the states
        self.optimizer = None
        if hasattr(self.accelerator, "deepspeed_engine_wrapped"):
            if self.accelerator.deepspeed_engine_wrapped is not None:
                self.accelerator.deepspeed_engine_wrapped.engine.empty_partition_cache()
                self.accelerator.deepspeed_engine_wrapped.engine.destroy()
            self.accelerator.deepspeed_engine_wrapped = None
        gc.collect()
        torch.cuda.empty_cache()
            
        # init new optimizer w/ new lisa layers
        self.create_optimizer()
        _, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        # put back non-lisa param groups
        self.optimizer.param_groups.extend([non_lisa_param_groups[0], non_lisa_param_groups[1]])
        if hasattr(self.accelerator, "deepspeed_engine_wrapped"):
            self._post_init_deepspeed_zero_optimizer_params(self.accelerator.deepspeed_engine_wrapped.engine.optimizer)
            
        # put back non-lisa states
        for gindex in NON_LISA_LAYER_PARAM_GROUPS_IDX:
            for param in self.optimizer.param_groups[gindex]['params']:
                self.optimizer.state[param] = non_lisa_states[param]
                
        del non_lisa_param_groups
        del non_lisa_states
        gc.collect()
        torch.cuda.empty_cache()
        
        if hasattr(self.accelerator, "deepspeed_engine_wrapped"):
            self.accelerator.deepspeed_engine_wrapped.engine.optimizer._link_all_hp_params()
        
        torch.cuda.memory._dump_snapshot(f'gs_{self.state.global_step}.pickle')
                
    
    def create_optimizer(self):
        """
        Setup the optimizer. Adopted from transformers.Trainer.create_optimizer.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            self._switch_active_layers() # init along with the optimizer
            
            optimizer_grouped_parameters = self._prepare_optimizer_param_group(opt_model)

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

        
        print(f"optim after create_optimizer {[len(pg['params']) for pg in self.optimizer.param_groups]=}")
        return self.optimizer
    
    
    def _prepare_optimizer_param_group(self, opt_model: nn.Module):
        decay_parameters = self.get_decay_parameter_names(opt_model)
        print(f"{decay_parameters=}")
        if not self._optimizer_param_group_initialized:
            optimizer_grouped_parameters = [
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
            ]
            self._optimizer_param_group_initialized = True
        else:
            optimizer_grouped_parameters = [
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
        
        return optimizer_grouped_parameters                    
        
        
    def _post_init_deepspeed_zero_optimizer_params(self, optimizer: DeepSpeedZeroOptimizer):
        optimizer.real_dp_process_group = [optimizer.dp_process_group for i in range(len(optimizer.optimizer.param_groups))]
        optimizer.partition_count = [dist.get_world_size(group=optimizer.dp_process_group) for i in range(len(optimizer.optimizer.param_groups))]
        
        for i, param_group in enumerate(optimizer.optimizer.param_groups):
            if i in LISA_BODY_LAYER_PARAM_GROUPS_IDX:
                # skip lisa layers
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
            
            
def tag(info=''):
    time.sleep(10)
    print(info)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)