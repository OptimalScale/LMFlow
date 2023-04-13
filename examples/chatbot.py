#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import sys
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


def rstrip_partial_utf8(string):
    return string.replace("\ufffd", "")


def print_to_console(string, encoding='utf-8', end="\n"):
    sys.stdout.buffer.write((string + end).encode(encoding))


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={
            "help": "prompt structure given user's input text"
        },
    )
    end_string: Optional[str] = field(
        default="\n\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )
    max_new_tokens: Optional[int] = field(
        default=200,
        metadata={
            "help": "maximum number of generated tokens"
        },
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "higher this value, more random the model output"
        },
    )


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
        device=pipeline_args.device,
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
    print_to_console(guide_message)

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = ""

    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure

    while True:
        input_text = input("User >>> ")
        if not input_text:
            print_to_console("exit...")
            break

        context += prompt_structure.format(input_text=input_text)
        context = context[-model.get_max_length():]     # Memory of the bot

        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": context } ]
        })

        print_index = 0
        response = ""

        token_per_step = 4
        for _ in range(0, chatbot_args.max_new_tokens // token_per_step):
            output_dataset = inferencer.inference(
                model=model,
                dataset=input_dataset,
                max_new_tokens=token_per_step,
                temperature=chatbot_args.temperature,
            )

            new_append_text = output_dataset.to_dict()["instances"][0]["text"]
            new_append_text = rstrip_partial_utf8(new_append_text)
            response += new_append_text

            input_dict = input_dataset.to_dict()
            input_dict["instances"][0]["text"] += new_append_text

            input_dataset = input_dataset.from_dict(input_dict)

            flag_break = False
            try:
                index = response.index(end_string)
                flag_break = True
            except ValueError:
                response += end_string
                index = response.index(end_string)

            response = response[:index]

            # Prints characters in the buffer
            new_print_index = print_index
            for char in response[print_index:]:
                if end_string is not None and char == end_string[0]:
                    if new_print_index + len(end_string) >= len(response):
                        break

                new_print_index += 1
                print_to_console(char, end="")

            print_index = new_print_index

            if flag_break:
                break
        print_to_console("\n", end="")

        context += response + "\n"


if __name__ == "__main__":
    main()
