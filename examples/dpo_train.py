#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/4/2024 20:31
# @Author  : Yu Li
# @Site    : 
# @File    : dpo_train.py
# 0. imports
import logging
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser, pipeline, AutoTokenizer

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

if __name__ == "__main__":
    # Parses arguments
    pipeline_name = "dpo_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
    parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        PipelineArguments,
    ))

    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Initializes pipeline, dataset and model for reward training
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    model = AutoModel.get_model(model_args)

    # Aligns model with rewards
    aligned_model = aligner.align(
        model=model,
        dataset=None,
        reward_model=None
    )