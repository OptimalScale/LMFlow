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
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    AutoArguments,
)
from lmflow.utils.common import remove_dataclass_attr_prefix, create_copied_dataclass


logger = logging.getLogger(__name__)


ReferenceModelArguments = create_copied_dataclass(
    original_dataclass=ModelArguments, 
    field_prefix="reference_",
    class_prefix="Reference"
)


def main():
    # Parses arguments
    pipeline_name = "dpov2_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments, 
        ReferenceModelArguments,
        DatasetArguments,
        PipelineArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, ref_model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, ref_model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
        
    ref_model_args_dict = remove_dataclass_attr_prefix(ref_model_args, "reference_")
    ref_model_args = ModelArguments(**ref_model_args_dict)

    train_dataset = Dataset(data_args)
    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.dataset_path = pipeline_args.eval_dataset_path
    eval_dataset = Dataset(eval_data_args)
    model = AutoModel.get_model(model_args)
    ref_model = AutoModel.get_model(ref_model_args)
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
        ref_model_args=ref_model_args,
    )

    res = aligner.align(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    

if __name__ == "__main__":
    main()