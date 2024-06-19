#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.

# Note that this is only a workaround, since vllm
# inference engine cannot release GPU memory properly by now. Please see this github 
# [issue](https://github.com/vllm-project/vllm/issues/1908).

import logging
import sys
import os
from typing import Dict

from transformers import (
    HfArgumentParser
)

from lmflow.datasets import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.vllm_inferencer import VLLMInferencer
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    AutoArguments,
)
from lmflow.utils.constants import MEMORY_SAFE_VLLM_INFERENCE_FINISH_FLAG


logger = logging.getLogger(__name__)


def main():
    # Parses arguments
    pipeline_name = "inferencer"
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
    model = HFDecoderModel(model_args)
    inferencer = VLLMInferencer(model_args, pipeline_args)

    res = inferencer.inference(
        model,
        dataset,
        release_gpu=False,
        detokenize=pipeline_args.memory_safe_vllm_inference_detokenize,
    )
    
    # use this as a flag, stdout will be captured by the pipeline
    print(MEMORY_SAFE_VLLM_INFERENCE_FINISH_FLAG)
    

if __name__ == "__main__":
    main()