#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import copy
import os
import torch
import wandb
import deepspeed
import sys
import numpy as np
import datetime
import json
import time
import logging
from typing import Dict, List, Union, Tuple, Any

from accelerate import Accelerator
import ray
import ray.data
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from tqdm import tqdm
from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import torch.distributed as dist
import torch.nn.functional as F

from lmflow.args import (
    DatasetArguments,
    ModelArguments,
    InferencerArguments,
)
from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.utils.data_utils import (
    set_random_seed,
    batchlize
)
from lmflow.datasets.dataset import KEY_SCORE


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)


class RewardModelInferencer(BasePipeline):
    """
    Initializes the `Inferencer` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    inferencer_args : InferencerArguments object.
        Contains the arguments required to perform inference.
    """
    def __init__(
        self, 
        model_args: ModelArguments, 
        data_args: DatasetArguments, 
        inferencer_args: InferencerArguments,
    ):
        self.data_args = data_args
        self.inferencer_args = inferencer_args
        self.model_args = model_args

        set_random_seed(self.inferencer_args.random_seed)

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if inferencer_args.device == "gpu":
            torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
            deepspeed.init_distributed()
        else:
            dist.init_process_group(
                "gloo", rank=self.local_rank, world_size=self.world_size
            )

        if inferencer_args.use_accelerator:
            self.accelerator = Accelerator()
            self.accelerator.wait_for_everyone()
            

    def inference(
        self,
        model: HFTextRegressionModel,
        dataset: Dataset,
        transform_dataset_in_place: bool=True,
        use_vllm: bool = False,
        enable_distributed_inference: bool = False,
        **kwargs,
    ) -> Dataset:
        if use_vllm:
            logger.warning("VLLM doesn't support reward model inference, using normal inference instead.")
            use_vllm = False
            
        assert isinstance(model, HFTextRegressionModel), "model should be HFTextRegressionModel"
        if not transform_dataset_in_place:
            dataset = copy.deepcopy(dataset)
            
        model_input = model.prepare_inputs_for_inference(
            dataset=dataset,
            apply_chat_template=True,
            enable_distributed_inference=enable_distributed_inference,
            use_vllm=use_vllm
        )
            
        if use_vllm:
            scores = self.__vllm_inference(
                model=model, 
                model_input=model_input,
                enable_distributed_inference=enable_distributed_inference,
            )
        else:
            scores = self._inference(
                model=model,
                model_input=model_input,
                enable_distributed_inference=enable_distributed_inference,
                **kwargs,
            )
            
        output_dataset = model.postprocess_inference_outputs(dataset, scores)
        
        return output_dataset
    
    
    def _inference(
        self,
        model: HFTextRegressionModel,
        model_input: Union[Dataset, ray.data.Dataset],
        enable_distributed_inference: bool = False,
        **kwargs,
    ):
        if enable_distributed_inference:
            inference_res = self.__inference(
                model=model, 
                model_input=model_input,
            )
        else:
            inference_res = self.__distributed_inference(
                model=model, 
                model_input=model_input, 
                num_instances=kwargs.get("distributed_inference_num_instances", 1),
                batch_size=kwargs.get("inference_batch_size", 1),
            )
        
        return inference_res


    def __inference(
        self,
        model: HFTextRegressionModel,
        model_input: Dataset,
    ) -> Union[List[float], List[List[float]]]:
        if model_input.get_type() in ["text_to_textlist"]:
            model_input_ids, num_outputs = self.flatten_list(model_input.get_backend_dataset()["input_ids"])
        else:
            model_input_ids = model_input.get_backend_dataset()["input_ids"]
            
        dataloader = batchlize(
            examples=model_input_ids,
            batch_size=self.inferencer_args.inference_batch_size,
            random_shuffle=False, # DO NOT shuffle when inference
        )
        num_batches = len(dataloader)
        final_output = []
        
        for batch_index, batched_input_ids in tqdm(
            iterable=enumerate(dataloader), 
            total=num_batches, 
            desc="Inference", 
            unit="batch"
        ):
            # len(batch) = batch_size, and batch element is dataset sample
            model_input_tensor = torch.LongTensor(batched_input_ids).to("cpu" if model.device == "cpu" else "cuda")
            if self.inferencer_args.use_accelerator:
                with self.accelerator.autocast():
                    batch_output = model.inference(
                        inputs=model_input_tensor, 
                        use_vllm=False,
                    )
            else:
                batch_output = model.inference(
                    inputs=model_input_tensor, 
                    use_vllm=False,
                )
            
            batch_output = self.__post_process_model_output(batch_output)
            final_output.extend(batch_output)
        
        if model_input.get_type() in ["text_to_textlist"]:
            final_output = self.compress_list(final_output, num_outputs)
        
        return final_output
    
    
    def __distributed_inference(
        self,
        model: HFTextRegressionModel,
        model_input: ray.data.Dataset,
        num_instances: int,
        batch_size: int,
    ):
        def scheduling_strategy_fn():
            # One bundle per tensor parallel worker
            pg = ray.util.placement_group(
                [{
                    "GPU": 1,
                    "CPU": 1
                }] * self.inferencer_args.tensor_parallel_size,
                strategy="STRICT_PACK",
            )
            return dict(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    pg, placement_group_capture_child_tasks=True
                )
            )
            
        resources_kwarg: Dict[str, Any] = {}
        if self.inferencer_args.vllm_tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            resources_kwarg["num_gpus"] = 1
        else:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            resources_kwarg["num_gpus"] = 0
            resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn
            
        ## predictor
        class DistributedPredictor:
            def __init__(
                self, 
                model: HFTextRegressionModel,
            ):
                self.model = copy.deepcopy(model)
                self.model.activate_model_for_inference(use_vllm=False)
                
            def __call__(self, batch: Dict[str, np.ndarray]):
                """batch: Dict[str, np.ndarray], {"item": array(['...', '...', '...', ...])}
                """
                batched_inference_res = self.model.inference(inputs=batch['item'])
                batched_final_res = {
                    "input": [sample['input'] for sample in batched_inference_res],
                    "output": [sample['output'] for sample in batched_inference_res] 
                } # do this since we're writing to a pandas dataframe
                return batched_final_res
    
    
    def __vllm_inference(
        self,
        model: HFTextRegressionModel,
        model_input: List[str],
        enable_distributed_inference: bool = False,
    ) -> List[float]:
        raise NotImplementedError("VLLM inference for reward model is not implemented yet.")
        
    
    def __post_process_model_output(
        self,
        model_output: SequenceClassifierOutputWithPast,
    ) -> List[float]:
        final_output = model_output.logits.to("cpu").reshape(-1).tolist()
        
        return final_output
            
    
    def flatten_list(
        self, 
        list_of_list: List[List]
    ) -> Tuple[List, List[int]]:
        sublist_lengths = [len(sublist) for sublist in list_of_list]
        flattened_list = [item for sublist in list_of_list for item in sublist]
        return flattened_list, sublist_lengths
    

    def compress_list(
        self, 
        list_to_compress: List, 
        sublist_lengths: List[int]
    ) -> List[List]:
        assert sum(sublist_lengths) == len(list_to_compress), "Sum of sublist lengths should be equal to length of list to compress."
        compressed_list = []
        start_index = 0
        for length in sublist_lengths:
            sublist = list_to_compress[start_index: start_index + length]
            compressed_list.append(sublist)
            start_index += length
        return compressed_list
