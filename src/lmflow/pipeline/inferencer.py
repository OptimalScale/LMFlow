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
import re

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
        if inferencer_args.device == "gpu":
            torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
            deepspeed.init_distributed()
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "15000"
            dist.init_process_group(
                "gloo", rank=self.local_rank, world_size=self.world_size
            )

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
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
        do_sample: bool=False,
        top_p: float=0.7,
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
            current_batch = batch[0]        # batch size is 1

            raw_input = prompt_structure.format(input=current_batch['input'])
            print(f"raw_input = {raw_input}")

            # if self.model_args.model_name_or_path == 'THUDM/chatglm-6b':
                # history = []
                # for response, history in model.get_backend_model().stream_chat(model.get_tokenizer(), raw_input, history=history):
                #     text_out = response

            # else:
            if self.inferencer_args.device == "gpu":
                inputs = model.encode(raw_input, return_tensors="pt").to(device=self.local_rank)
            elif self.inferencer_args.device == "cpu":
                inputs = model.encode(raw_input, return_tensors="pt").to(device='cpu')
            else:
                raise NotImplementedError(
                    f"device \"{self.inferencer_args.device}\" is not supported"
                )

            outputs = model.inference(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=1.0,
                do_sample=do_sample,
                top_p=top_p,
                raw_input=raw_input,
            )
            text_out = model.decode(outputs[0], skip_special_tokens=True)
            print(f"text_out = {text_out}")
            flag_break = False
            if model.get_tokenizer().eos_token_id in outputs[0]:
                flag_break = True

            # only return the generation, trucating the input
            prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
            text_out = text_out[prompt_length:]
            output_dict["instances"].append({ "text": text_out })

        output_dataset = Dataset(DatasetArguments(dataset_path = None))
        output_dataset = output_dataset.from_dict(output_dict)

        print(f"output_dataset = {output_dataset}")
        print(f"type = {type(output_dataset)}")
        return output_dataset, flag_break
