import os
import deepspeed
import torch
import wandb
import sys
import numpy as np
import datetime
import json
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
# TODO: remove later
from transformers import AutoConfig
from lmflow.pipeline.auto_pipeline import AutoPipeline

import torch.distributed as dist
from transformers import HfArgumentParser
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.utils.data_utils import set_random_seed, batchlize, answer_extraction, load_data
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# copied from evaluate.py, with small changes.
pipeline_name = "test_rougel"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)
dataset = Dataset(data_args)

evaluator = AutoPipeline.get_pipeline(
    pipeline_name=pipeline_name,
    model_args=model_args,
    data_args=data_args,
    pipeline_args=pipeline_args,
)
evaluator.evaluate(model=model, dataset=dataset, metric=pipeline_args.metric)
