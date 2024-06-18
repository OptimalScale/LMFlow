#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from typing import Dict

import torch
from transformers import (
    HfArgumentParser
)

from lmflow.args import (
    ModelArguments,
    OnlineRLHFDatasetArguments,
    AutoArguments,
)
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.utils.collections import create_copied_dataclass, remove_dataclass_attr_prefix


logger = logging.getLogger(__name__)


RewardModelArguments = create_copied_dataclass(
    original_dataclass=ModelArguments, 
    field_prefix="reward_", 
    class_prefix="Reward", 
    new_default={
        "reward_arch_type":"text_regression"
    }
)


def main():
	# Parses arguments
    pipeline_name = "iterative_dpo_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments, 
        RewardModelArguments, 
        OnlineRLHFDatasetArguments, 
        PipelineArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, reward_model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, reward_model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
        
    reward_model_args = ModelArguments(**remove_dataclass_attr_prefix(reward_model_args, "reward_"))

    # Initialization
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        reward_model_args=reward_model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Align (init models, data inside aligner since they will change during an online training process)
    tuned_model = aligner.align(model_args=model_args, data_args=data_args, reward_model_args=reward_model_args)


if __name__ == '__main__':
    main()
