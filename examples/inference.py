#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
    ))
    model_args, pipeline_args = parser.parse_args_into_dataclasses()
    inferencer_args = pipeline_args

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
        use_accelerator=True,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Inferences
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

    while True:
        input_text = input("User >>> ")
        input_text = input_text[-model.get_max_length():]     # Truncation

        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": input_text } ]
        })

        output_dataset = inferencer.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=inferencer_args.max_new_tokens,
            temperature=inferencer_args.temperature,
        )
        output = output_dataset.to_dict()["instances"][0]["text"]
        print(output)


if __name__ == "__main__":
    main()
