#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
# FIXME should merge with finetune.py
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.args import (
    VisModelArguments,
    MultiModalDatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

from lmflow.models.vision2seq_model import CustomAutoVision2SeqModel
from lmflow.models.vision_encoder import build_vision_tower
from lmflow.datasets.multi_modal_dataset import DataCollatorForSupervisedDataset
from torch.utils.data import DataLoader




def main():
	# Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((VisModelArguments, MultiModalDatasetArguments, PipelineArguments))
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
    model = AutoModel.get_model(model_args, tune_strategy='none',
                                ds_config=pipeline_args.deepspeed,
                                custom_model=True)
    for param in model.backend_model.language_model.parameters():
        param.requires_grad=False
    for param in model.backend_model.vision_model.parameters():
        param.requires_grad=False
    dataset = Dataset(data_args, backend="custom_multi_modal")
    # dataset.backend_dataset.register_tokenizer(model.tokenizer, model.image_processor)
    # dataset.backend_dataset.__getitem__(0)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=model.tokenizer)

    # Finetuning
    tuned_model = finetuner.tune(
        model=model, dataset=dataset, data_collator=data_collator)


if __name__ == '__main__':
    main()
