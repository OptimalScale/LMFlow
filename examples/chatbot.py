#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
import torch.distributed as dist

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class ChatbotArguments:
    pass


def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
        ChatbotArguments,
    ))
    model_args, pipeline_args, chatbot_args = (
        parser.parse_args_into_dataclasses()
    )

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

    guide_message = (
        "\n"
        f"#############################################################################\n"
        f"##   A {model_name} chatbot is now chatting with you!\n"
        f"#############################################################################\n"
        "\n"
    )
    print(guide_message, end="")

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = ""
    end_string = "\n\n"

    while True:
        if dist.get_rank() == 0:
            input_text = input("User >>> ")            
            dist.broadcast_object_list([input_text])
        else:
            recev_object = [None] * 1
            dist.broadcast_object_list([recev_object])
            input_text = recev_object[0]

        if not input_text:
            print("exit...")
            break

        context += input_text

        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": context } ]
        })

        output_dataset = inferencer.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=200,
            temperature=0.7,
        )

        response = output_dataset.to_dict()["instances"][0]["text"]

        try:
            index = response.index(end_string)
        except ValueError:
            response += end_string
            index = response.index(end_string)

        response = response[:index + 1]
        print("Bot: " + response, end="")
        context += response
        context = context[-model.get_max_length():]     # Memory of the bot


if __name__ == "__main__":
    main()
