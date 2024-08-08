#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import os
import sys

from transformers import (
    HfArgumentParser
)

from lmflow.datasets import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    AutoArguments,
)


logger = logging.getLogger(__name__)


def main():
    # Parses arguments
    pipeline_name = "vllm_inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments, 
        DatasetArguments,
        PipelineArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args, tune_strategy='none')
    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args
    )

    res = inferencer.inference(
        model,
        dataset,
        release_gpu=False,
        enable_decode_inference_result=pipeline_args.enable_decode_inference_result,
        enable_distributed_vllm_inference=pipeline_args.enable_distributed_vllm_inference,
    )
    

if __name__ == "__main__":
    main()