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
    batchlize,
    RewardModelInferenceResultWithInput,
)
from lmflow.datasets.dataset import KEY_SCORE
from lmflow.utils.versioning import is_ray_available

if is_ray_available():
    import ray
    import ray.data
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


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
        **kwargs,
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
            self.accelerator: Accelerator = kwargs.get('accelerator', Accelerator())
            

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
            inference_result = self.__vllm_inference(
                model=model, 
                model_input=model_input,
                enable_distributed_inference=enable_distributed_inference,
            )
        else:
            inference_result = self._inference(
                model=model,
                model_input=model_input,
                enable_distributed_inference=enable_distributed_inference,
                **kwargs,
            )
        
        if enable_distributed_inference:
            output_dataset = model.postprocess_distributed_inference_outputs(
                dataset=dataset,
                inference_result=inference_result,
            )
        else:
            output_dataset = model.postprocess_inference_outputs(
                dataset=dataset, 
                scores=inference_result
            )
        
        return output_dataset
    
    
    def _inference(
        self,
        model: HFTextRegressionModel,
        model_input: Union[Dataset, 'ray.data.Dataset'],
        enable_distributed_inference: bool = False,
        **kwargs,
    ):
        if enable_distributed_inference:
            if not is_ray_available():
                raise ImportError('Ray is not installed. Please install via `pip install -e ".[ray]"`.')
            
            inference_res = self.__distributed_inference(
                model=model, 
                model_input=model_input, 
                num_instances=kwargs.get("distributed_inference_num_instances", 1),
                batch_size=kwargs.get("inference_batch_size", 1),
            )
        else:
            inference_res = self.__inference(
                model=model, 
                model_input=model_input,
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
        model_input: 'ray.data.Dataset',
        num_instances: int,
        batch_size: int,
    ) -> List[RewardModelInferenceResultWithInput]:
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
        if self.inferencer_args.tensor_parallel_size == 1:
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
                model_args: ModelArguments,
            ):
                self.model = HFTextRegressionModel(
                    model_args=model_args, 
                    tune_strategy='none', 
                    use_accelerator=True
                )
                self.model.activate_model_for_inference(use_vllm=False)
                
            def __call__(self, batch: Dict[str, np.ndarray]):
                """batch: Dict[str, np.ndarray]
                Example (batch size=2):
                {'input': array(['...','...'], dtype=object),
                 'output': array([array(["...", "..."], dtype=object), array(['...','...'], dtype=object)], dtype=object),
                 'input_ids': array([[[128000, 128006,    882, ..., 128256, 128256, 128256],
                         [128000, 128006,    882, ..., 128256, 128256, 128256]],
                        [[128000, 128006,    882, ..., 128256, 128256, 128256],
                         [128000, 128006,    882, ..., 128256, 128256, 128256]]])}
                """
                # The batch is managed by ray and the actual batch size may smaller than 
                # inference_batch_size in config, since there may be some remainders. 
                # For example, 10 examples with 2 inference instances and inference_batch_size=4,
                # there will be only 2 examples for instance 0 to run and then the 
                # actual batch size changes.
                actual_batch_size = len(batch['input'])
                input_tensor = torch.LongTensor([
                    [list(arr) for arr in batch['input_ids'][batch_idx]] 
                    for batch_idx in range(actual_batch_size)
                ]).flatten(start_dim=0, end_dim=1).to("cuda")
                batched_inference_res = self.model.inference(input_tensor).logits
                batched_inference_res = batched_inference_res.to("cpu").reshape(actual_batch_size, -1, 1).squeeze(dim=-1).tolist() 
                # [bs, num_output_sequences]
                batched_final_res = {
                    "input": batch['input'].tolist(),
                    "output": [
                        [
                            {"score": batched_inference_res[j][i], "text": batch["output"][j][i]}
                            for i in range(len(batch['output'][j]))
                        ] 
                        for j in range(actual_batch_size)
                    ],
                } # do this since we're writing to a pandas dataframe
                return batched_final_res

        # inference
        model_input_mapping = model_input.map_batches(
            DistributedPredictor,
            concurrency=num_instances, # Set the concurrency to the number of LLM instances.
            batch_size=batch_size,
            fn_constructor_kwargs={
                "model_args": model.model_args,
            },
            **resources_kwarg,
        )
        
        df_model_output = model_input_mapping.to_pandas() # the actual forwards are executed here
        logger.info(f"Distributed reward model inference result preview:\n{df_model_output.head(10)}")
        
        model_output = [
            {"input": row["input"], "output": row["output"]} for _, row in df_model_output[:].iterrows()
        ]
        
        return model_output
    
    
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
