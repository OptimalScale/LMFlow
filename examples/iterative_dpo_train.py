#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import os
import sys
import copy

from transformers import (
    HfArgumentParser
)

from lmflow.datasets import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    AutoArguments,
)
from lmflow.utils.common import remove_dataclass_attr_prefix, create_copied_dataclass


logger = logging.getLogger(__name__)


# NOTE:
# In training processes that needs more than one model such as dpo (reference & target),
# ppo (actor & critic), etc., we use the following function to create separate model arguments 
# to distinguish among them.
ReferenceModelArguments = create_copied_dataclass(
    original_dataclass=ModelArguments, 
    field_prefix="reference_",
    class_prefix="Reference"
)

RewardModelArguments = create_copied_dataclass(
    original_dataclass=ModelArguments, 
    field_prefix="reward_",
    class_prefix="Reward"
)


def main():
    pipeline_name = "iterative_dpo_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments, 
        ReferenceModelArguments,
        RewardModelArguments,
        DatasetArguments,
        PipelineArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, ref_model_args, reward_model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, ref_model_args, reward_model_args, data_args, pipeline_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, ref_model_args, reward_model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
        
    ref_model_args_dict = remove_dataclass_attr_prefix(ref_model_args, "reference_")
    ref_model_args = ModelArguments(**ref_model_args_dict)
    reward_model_args_dict = remove_dataclass_attr_prefix(reward_model_args, "reward_")
    reward_model_args = ModelArguments(**reward_model_args_dict)

    dataset_list = []
    for dataset in pipeline_args.dataset_path_list:
        iter_data_args = copy.deepcopy(data_args)
        iter_data_args.dataset_path = dataset
        dataset_list.append(Dataset(iter_data_args))
    
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
        ref_model_args=ref_model_args,
        reward_model_args=reward_model_args,
    )

    aligner.align(dataset_list=dataset_list)
    

if __name__ == "__main__":
    main()