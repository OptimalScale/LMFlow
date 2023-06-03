#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""Alignment tuning example, such as RLHF."""

import logging
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, pipeline, AutoTokenizer

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


@dataclass
class RewardArguments:
    reward_type: Optional[str] = field(
        default="hf_pipeline",
        metadata={
            "help": (
                "type of reward model, support huggingface pipeline. Will"
                " support \"customized\" torch.nn.modules in the future."
            ),
        },
    )
    reward_model_or_path: Optional[str] = field(
        default="weqweasdas/hh_rlhf_rm",
        metadata={
            "help": (
                "reward model name (huggingface) or its path"
            ),
        },
    )
    reward_task: Optional[str] = field(
        default="sentiment-analysis",
        metadata={
            "help": "type of reward task, such as sentiment-analysis, detoxic."
        },
    )
    reward_model_args: Optional[str] = field(
        default="return_all_scores=True, function_to_apply=\"none\", batch_size=1",
        metadata={
            "help": (
                "extra arguments required by different type of reward models."
            ),
        },
    )


def get_reward_function(reward_args, pipeline_args):
    args = reward_args
    reward_type = args.reward_type

    if reward_type == "hf_pipeline":

        # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
        # only for this model.
        rm_tokenizer = AutoTokenizer.from_pretrained(reward_args.reward_model_or_path)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
        rm_tokenizer.padding_side = "left"
        
        hf_pipe = pipeline(
            reward_args.reward_task,
            model=reward_args.reward_model_or_path,
            device=f"cuda:{pipeline_args.local_rank}",
            tokenizer=rm_tokenizer
        )
        def reward_func(dataset: Dataset):
            if dataset.type != "text_only":
                raise NotImplementedError(
                    "reward function only accept \"text_only\" datasets"
                )
            pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 1
            }

            data_dict = dataset.to_dict()
            texts_for_rewards = [
                sample["text"] for sample in data_dict["instances"]
            ]
            pipe_outputs = hf_pipe(texts_for_rewards, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]

            reward_dataset = Dataset.create_from_dict({
                "type": "float_only",
                "instances": [
                    { "value": reward } for reward in rewards
                ]
            })
            return reward_dataset

        return reward_func
    else:
        raise NotImplementedError("unsupported reward type \"{reward_type}\"")


def main():
	# Parses arguments
    pipeline_name = "raft_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        PipelineArguments,
        RewardArguments,
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args, reward_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args, reward_args = parser.parse_args_into_dataclasses()

    # Initializes pipeline, dataset and model for reward training
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)

    # Initializes reward function
    reward_function = get_reward_function(reward_args, pipeline_args)

    reward_model_args = ModelArguments(arch_type="text_regression")
    reward_model = AutoModel.get_model(reward_model_args)
    reward_model.register_inference_function(reward_function)

    # Aligns model with rewards
    aligned_model = aligner.align(
        model=model,
        dataset=dataset,
        reward_model=reward_model,
    )


if __name__ == '__main__':
    main()