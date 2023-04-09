#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import os
import sys

from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
	# Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # TODO: deepspeed config initialization

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)

    model = AutoModel.get_model(
        model_args,
        lang=data_args.lang,
        forced_bos_token=data_args.forced_bos_token,
        source_prefix = data_args.source_prefix,
        streaming = data_args.streaming,
        preprocessing_num_workers = data_args.preprocessing_num_workers,
        overwrite_cache = data_args.overwrite_cache,
        max_source_length = data_args.max_source_length,
        max_target_length = data_args.max_target_length,
        pad_to_max_length = data_args.pad_to_max_length
    )

    # Tokenization and text grouping must be done in the main process
    with pipeline_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = model.tokenize(dataset)
        if model_args.arch_type == "encoder_decoder": 
            # encoder-decoder model does not need group text
            lm_dataset = tokenized_dataset
        else:
            lm_dataset = finetuner.group_text(
                tokenized_dataset,
                model_max_length=model.get_max_length(),
            )

    # Finetuning
    tuned_model = finetuner.tune(model=model, lm_dataset=lm_dataset)


if __name__ == '__main__':
    main()
