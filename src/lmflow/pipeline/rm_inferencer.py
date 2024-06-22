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
from typing import Dict, List, Union

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
    batchlize
)


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
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "15000"
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
    ) -> Dataset:
        if use_vllm:
            logger.warning("VLLM doesn't support reward model inference, using normal inference instead.")
            use_vllm = False
            
        assert isinstance(model, HFTextRegressionModel), "model should be HFTextRegressionModel"
        if not transform_dataset_in_place:
            dataset = copy.deepcopy(dataset)
        output_dict = {
            "type": f"scored_{dataset.get_type()}",
            "instances": dataset.to_dict()["instances"],
        }
        
        if use_vllm:
            scores = self.__vllm_inference(model, dataset)
        else:
            scores = self.__inference(model, dataset)
            
        for i, score in enumerate(scores):
            output_dict["instances"][i]["score"] = score
        
        output_dataset_args = copy.deepcopy(self.data_args)
        output_dataset_args.dataset_path = None
        output_dataset_args.dataset_name = f"scored_{output_dataset_args.dataset_name}"
        output_dataset = Dataset(output_dataset_args)
        output_dataset = output_dataset.from_dict(output_dict)
        
        return output_dataset


    def __inference(
        self,
        model: HFTextRegressionModel,
        dataset: Dataset,
    ) -> List[float]:
        tokenized_dataset = model.tokenize(dataset)
        dataloader, _ = self.create_dataloader(
            dataset=tokenized_dataset,
            batch_size=self.inferencer_args.inference_batch_size,
            random_shuffle=False, # no need to shuffle when inference
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
            model_input = torch.LongTensor(batched_input_ids).to("cpu" if model.device == "cpu" else "cuda")
            if self.inferencer_args.use_accelerator:
                with self.accelerator.autocast():
                    batch_output = model.inference(
                        inputs=model_input, 
                        use_vllm=False,
                    )
            else:
                batch_output = model.inference(
                    inputs=model_input, 
                    use_vllm=False,
                )
            
            batch_output = self.__post_process_model_output(batch_output)
            final_output.extend(batch_output)
        
        return final_output
    
    
    def __vllm_inference(
        self,
        model: HFTextRegressionModel,
        dataset: Dataset,
    ) -> List[float]:
        raise NotImplementedError("VLLM inference for reward model is not implemented yet.")
        
    
    def __post_process_model_output(
        self,
        model_output: SequenceClassifierOutputWithPast,
    ) -> List[float]:
        final_output = model_output.logits.to("cpu").reshape(-1).tolist()
        
        return final_output
        
        
    def create_dataloader(
        self, 
        dataset: Dataset,
        batch_size: int = 1,
        random_shuffle: bool = False,
    ):
        r"""Batchlize dataset and format it to dataloader.

        Args:
            dataset (Dataset): the dataset object

        Output:
            dataloader (batchlize): the dataloader object
            dataset_size (int): the length of the dataset

        """
        inputs = dataset.get_backend_dataset()["input_ids"] # this comes from lmflow model.tokenize(dataset)
        dataset_size = len(inputs)

        dataloader = batchlize(
            inputs,
            batch_size=batch_size,
            random_shuffle=random_shuffle,
        )
        return dataloader, dataset_size
    