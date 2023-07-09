#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell to inference the input data.
"""
from dataclasses import dataclass, field
import logging
import json
import requests
from PIL import Image
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from typing import Optional
import warnings

from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import (ModelArguments, DatasetArguments, \
                            InferencerArguments, AutoArguments)

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={
            "help": "prompt structure given user's input text"
        },
    )
    end_string: Optional[str] = field(
        default="\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )
    image_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "image path for input image"}
    )
    input_text: Optional[str] = field(
        default="",
        metadata={
            "help": "input text for reasoning"}
    )
    task: Optional[str] = field(
        default="image_caption",
        metadata={
            "help": "task for reasoning",
        }
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
    inferencer_args = pipeline_args
    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
    )

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
    print(guide_message)

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = ""

    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure


    # Load image and input text for reasoning
    if chatbot_args.image_path is not None:
        raw_image = Image.open(chatbot_args.image_path)
    else:
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    input_text = chatbot_args.input_text
    if chatbot_args.task == "image_caption" and len(input_text) == 0:
        input_text = "a photography of"


    if chatbot_args.task == "image_caption":
        # single round reasoning
        input_dataset = dataset.from_dict({
            "type": "image_text",
            "instances": [{"images": raw_image,
                        "text":  input_text,}]
        })
        output = inferencer.inference(model, input_dataset)
        print(output.backend_dataset['text'])
    else:
        # multi rounds reasoning
        # TODO support streaming reasoning.
        while True:
            input_text = input("User >>> ")
            if input_text == "exit":
                print("exit...")
                break
            elif input_text == "reset":
                context = ""
                print("Chat history cleared")
                continue
            if not input_text:
                input_text = " "
            context += prompt_structure.format(input_text=input_text)
            # TODO handle when model doesn't have the get_max_length
            context = context[-model.get_max_length():]     # Memory of the bot
            input_dataset = dataset.from_dict({
                "type": "image_text",
                "instances": [{"images": raw_image,
                            "text":  context,}]
            })

            print("Bot: ", end="")
            print_index = 0

            token_per_step = 4

            for response, flag_break in inferencer.stream_inference(
                context=context,
                model=model,
                max_new_tokens=inferencer_args.max_new_tokens,
                token_per_step=token_per_step,
                temperature=inferencer_args.temperature,
                end_string=end_string,
                input_dataset=input_dataset
            ):
                # Prints characters in the buffer
                new_print_index = print_index
                for char in response[print_index:]:
                    if end_string is not None and char == end_string[0]:
                        if new_print_index + len(end_string) >= len(response):
                            break

                    new_print_index += 1
                    print(char, end="", flush=True)

                print_index = new_print_index

                if flag_break:
                    break
            print("\n", end="")

            context += response + "\n"


if __name__ == "__main__":
    main()
