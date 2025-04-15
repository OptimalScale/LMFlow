#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
The finetuning of each epoch will be splitted into `--num_stages_per_epoch`
stages, where each stage will restart finetuning and continuously train from
the finetuned model of previous stage.

For example, with `--num_train_epochs 3 --num_stages_per_epoch 4`, there will
be totally 4*3 stages, and the learning rate schedule will restart every stage.
"""

import logging
import random
import sys
import os
import gc
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from dataclasses import dataclass, field

from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


logger = logging.getLogger(__name__)


@dataclass
class MultistageFinetuneArgs:
    num_stages_per_epoch: int = field(
        default=1,
        metadata={"help": "number of stages per epoch"}
    )
    shuffle_base_seed: int = field(
        default=23,
        metadata={
            "help": "base seed for generating dataset shuffle seeds each epoch"
        }
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "start from a specific epoch"}
    )


def setup_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def generate_new_seed(seed):
    return (seed * 37) % 1000000009


def shuffle_and_split_data(dataset, num_split=None, seed=None):
    data_dict = dataset.to_dict()

    random.seed(seed)
    random.shuffle(data_dict["instances"])

    dataset_list = []
    for i in range(num_split):
        partial_data_dict = {
            "type": data_dict["type"],
            "instances": data_dict["instances"][i::num_split]
        }
        partial_dataset = Dataset.create_from_dict(partial_data_dict)
        dataset_list.append(partial_dataset)

    return dataset_list


def main():
    # Initializes logger
    setup_logger()

	# Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        PipelineArguments,
        MultistageFinetuneArgs,
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a
        # json file, let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args, multistage_args = parser.parse_args_into_dataclasses()

    # Initialization
    full_dataset = Dataset(data_args)

    finetuner_args = pipeline_args
    logger.warning(
        "force set `--overwrite_output_dir True`"
        " as required by multistage finetuning"
    )
    finetuner_args.overwrite_output_dir = True

    # Finetuning
    num_train_epochs = finetuner_args.num_train_epochs
    if abs(num_train_epochs - int(num_train_epochs + 0.5)) > 1e-6:
        raise ValueError("only support int-typed `--num_train_epochs`")

    output_dir = finetuner_args.output_dir
    run_name = finetuner_args.run_name
    finetuner_args.num_train_epochs = 1      # Finetune every 1 epoch
    shuffle_seed = multistage_args.shuffle_base_seed

    for epoch in range(int(num_train_epochs)):
        shuffle_seed = generate_new_seed(shuffle_seed)
        dataset_list = shuffle_and_split_data(
            full_dataset,
            num_split=multistage_args.num_stages_per_epoch,
            seed=shuffle_seed,
        )
        if epoch < multistage_args.start_epoch:
            logging.info(f"skip epoch {epoch}")
            continue

        for stage, dataset in enumerate(dataset_list):
            is_main_process = (finetuner_args.local_process_index == 0)
            if is_main_process:
                logger.setLevel(logging.INFO)
                logger.info(f"========== epoch {epoch} stage {stage} ==========")

            # Initialization for each sub-finetune
            model = AutoModel.get_model(model_args)
            finetuner_args.output_dir = (
                output_dir + f"_epoch-{epoch}_stage-{stage}"
            )
            finetuner_args.run_name = run_name + f"_epoch-{epoch}_stage-{stage}"

            finetuner = AutoPipeline.get_pipeline(
                pipeline_name=pipeline_name,
                model_args=model_args,
                data_args=data_args,
                pipeline_args=finetuner_args,
            )

            # Finetunes on piece of data
            tuned_model = finetuner.tune(
                model=model,
                dataset=dataset,
                transform_dataset_in_place=False,
            )
            model_args.model_name_or_path = finetuner_args.output_dir
            del model
            del tuned_model
            del finetuner
            tuned_model = None
            model = None
            finetuner = None
            gc.collect()


    tuned_model.save(output_dir)


if __name__ == '__main__':
    main()
