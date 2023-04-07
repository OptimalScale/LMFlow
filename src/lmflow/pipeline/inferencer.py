#!/usr/bin/env python
# coding=utf-8
"""The Inferencer class simplifies the process of model inferencing."""

import os
import torch
import wandb
import deepspeed
import sys
import numpy as np
import datetime
import json

from transformers import AutoConfig
import torch.distributed as dist

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.utils.data_utils import set_random_seed, batchlize, answer_extraction
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

class Inferencer(BasePipeline):
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
    def __init__(self, model_args, data_args, inferencer_args):
        self.data_args = data_args
        self.inferencer_args = inferencer_args
        self.model_args = model_args

        set_random_seed(self.inferencer_args.random_seed)

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
        deepspeed.init_distributed()

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        try: 
            self.model_hidden_size = self.config.hidden_size
        except:
            print("Error in setting hidden size, use the default size 1024")
            self.model_hidden_size = 1024 # gpt2 seems do not have hidden_size in config


    def create_dataloader(self, dataset: Dataset):
        data_dict = dataset.to_dict()
        inputs = [ instance["text"] for instance in data_dict["instances"] ]
        dataset_size = len(inputs)
        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({
                "input": inputs[idx],
                "input_idx": idx
            })

        dataloader = batchlize(
            dataset_buf,
            batch_size=1,
            random_shuffle=False,
        )
        return dataloader, dataset_size


    def inference(
        self,
        model,
        dataset: Dataset,
        max_new_tokens: int=100,
        temperature: float=0.0,
        prompt_structure: str='{input}',
    ):
        """
        Perform inference for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform inference

        dataset : Dataset object.
            

        Returns:

        output_dataset: Dataset object.
        """
        if dataset.get_type() != "text_only":
            raise NotImplementedError(
                'input dataset should have type "text_only"'
            )

        dataloader, data_size = self.create_dataloader(dataset)

        # The output dataset
        output_dict = {
            "type": "text_only",
            "instances": [
            ]
        }

        for batch_index, batch in enumerate(dataloader):
            current_batch = batch[0]

            input = prompt_structure.format(input=current_batch['input'])
            
            inputs = model.encode(input, return_tensors="pt").to(device=self.local_rank)
            mask = inputs['attention_mask'].to(device=self.local_rank)
            outputs = model.inference(
                inputs,
                max_new_tokens=max_new_tokens,
                attenion_mask=mask,
                temperature=temperature
            )
            text_out = model.decode(outputs, skip_special_tokens=True)[0]

            # only return the generation, trucating the input
            prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
            text_out = [text_out[i][prompt_length[i]:] for i in range(len(text_out))]
            output_dict["instances"].append({ "text": text_out })

        output_dataset = Dataset(DatasetArguments(dataset_path = None))
        output_dataset = output_dataset.from_dict(output_dict)

        return output_dataset
