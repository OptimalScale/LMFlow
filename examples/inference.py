#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell to inference the input data.
"""
import logging
import json
import requests
from PIL import Image
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings

from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import (ModelArguments, DatasetArguments, \
                            InferencerArguments, AutoArguments)

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
    ))

    model_args, pipeline_args = (
        parser.parse_args_into_dataclasses()
    )
    inferencer_args = pipeline_args
    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
    )

    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    # Load image and input text for reasoning
    if inferencer_args.image_path is not None:
        raw_image = Image.open(inferencer_args.image_path)
    else:
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    input_text = inferencer_args.input_text
    if inferencer_args.task == "image_caption" and len(input_text) == 0:
        input_text = "a photography of"

    input_dataset = dataset.from_dict({
        "type": "image_text",
        "instances": [{"images": raw_image,
                       "text":  input_text,}]
    })

    # TODO support different prompt text
    prompt_text = ""
    output = inferencer.inference(model, input_dataset,
                    prompt_structure=prompt_text + "{input}")
    print(output.backend_dataset['text'])

if __name__ == "__main__":
    main()
