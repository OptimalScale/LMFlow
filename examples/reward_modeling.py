#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import os
import sys

sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from transformers import HfArgumentParser

from lmflow.args import (
    AutoArguments,
    DatasetArguments,
    ModelArguments,
)
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

logger = logging.getLogger(__name__)


def main():
    # Parses arguments
    pipeline_name = "rm_tuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)

    # Finetuning
    finetuner.tune(model=model, dataset=dataset)


if __name__ == "__main__":
    main()
