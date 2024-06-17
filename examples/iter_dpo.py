#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from dataclasses import make_dataclass, fields, Field
from typing import Dict

import torch
from transformers import (
    HfArgumentParser
)

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


logger = logging.getLogger(__name__)


def create_copied_dataclass(
    original_dataclass, 
    field_prefix: str, 
    class_prefix: str, 
    new_default: Dict=None
):
    """Create a copied dataclass with new field names and default values.

    Parameters
    ----------
    original_dataclass : dataclass
    field_prefix : str
        The prefix to add to the **field** names of the copied dataclass.
    class_prefix : str
        The prefix to add to the **class** name of the copied dataclass.
    new_default : Dict, optional
        The new default values for the copied dataclass. When None, the 
        default values of the original dataclass are used.

    Returns
    -------
    dataclass
    """
    original_fields = fields(original_dataclass)
    new_default = new_default or {}
    new_fields = []
    for field in original_fields:
        new_field = (
            f"{field_prefix}{field.name}", 
            field.type, 
            Field(
                default=new_default.get(f"{field_prefix}{field.name}", field.default), 
                default_factory=field.default_factory,
                init=field.init,
                repr=field.repr,
                hash=field.hash,
                compare=field.compare,
                metadata=field.metadata,
            )
        )
        new_fields.append(new_field)
    copied_dataclass = make_dataclass(f"{class_prefix}{original_dataclass.__name__}", new_fields)
    return copied_dataclass


def remove_dataclass_attr_prefix(data_instance, prefix: str) -> Dict:
    """Remove the prefix from the attribute names of a dataclass instance.

    Parameters
    ----------
    data_instance : dataclass
    prefix : str
        The prefix to remove from the attribute names of the dataclass instance.

    Returns
    -------
    Dict
    """
    new_attributes = {}
    for field in fields(data_instance):
        attr_name = field.name
        attr_value = getattr(data_instance, attr_name)
        new_attr_name = f"{attr_name[len(prefix):]}"
        new_attributes[new_attr_name] = attr_value
    
    return new_attributes


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
    pipeline_name = "iter_dpo_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, RewardModelArguments, DatasetArguments, PipelineArguments))
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
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)
    reward_model = AutoModel.get_model(reward_model_args)
        
    # Align
    tuned_model = aligner.align(model=model, dataset=dataset, reward_model=reward_model)


if __name__ == '__main__':
    main()
