#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

import torch
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.pipeline.utils.reward_dataprocessor import (
    build_dataset,
    RewardDataCollatorWithPadding
)


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
    
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        num_labels=1, 
        torch_dtype=torch.bfloat16, 
        use_flash_attention_2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        use_auth_token=True
    )
    tokenizer.truncation_side = pipeline_args.truncation_side
    tokenizer.model_max_length = pipeline_args.model_max_length
        
    data_collator = RewardDataCollatorWithPadding(
        tokenizer=tokenizer, 
        max_length=pipeline_args.model_max_length
    )
    train_dataset, eval_dataset = build_dataset(
        tokenizer=tokenizer, 
        train_path=pipeline_args.train_dataset_path,
        eval_path=pipeline_args.eval_dataset_path
    )
    logger.warning(f"Training set: {len(train_dataset)}, Eval set: {len(eval_dataset)}")
    
    # Finetuning
    tuned_model = finetuner.tune(
        model=model, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator)


if __name__ == '__main__':
    main()
