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
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

def write_json(data,path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

pipeline_name = "inferencer"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(
    model_args, 
    tune_strategy='none', 
    ds_config=ds_config, 
    use_accelerator=pipeline_args.use_accelerator_for_evaluator
)
dataset = Dataset(data_args)

inferencer = AutoPipeline.get_pipeline(
    pipeline_name=pipeline_name,
    model_args=model_args,
    data_args=data_args,
    pipeline_args=pipeline_args,
)
output_datasets, output_file = inferencer.inference(model=model, dataset=dataset, max_new_tokens=512, temperature=0.7)
write_json(output_file, pipeline_args.output_result_path+'/results.json')
