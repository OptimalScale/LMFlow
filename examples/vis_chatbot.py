#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell to inference the input data.
"""
from cmath import e
from dataclasses import dataclass, field
import logging
import json
import numpy as np
import requests
from PIL import Image
import os
import os.path as osp
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import time
from typing import Optional
import warnings

from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import (VisModelArguments, DatasetArguments, \
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
        default="#",
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
        default="vqa",
        metadata={
            "help": (
                "task for reasoning"
                "If do the caption task, the input text is describe "
                "the image and the conversation is only one round"
                "If other, the conversation is multi-round"
            )
        }
    )
    chatbot_type: Optional[str] = field(
        default="None",
        metadata={
            "help": (
                "prompt format"
                "the default format is ''"
                "Anthoer format is they way in mini-gpt4."
            )
        }
    )
    stream_inference: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to do the stream inference"
        }
    )
    with_deepspeed: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to use deepspeed"
        }
    )

def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        VisModelArguments,
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
        custom_model=model_args.custom_model,
        with_deepspeed=chatbot_args.with_deepspeed,
    )

    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args, backend="dict")

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.llm_model_name_or_path is not None:
        model_name = model_name + " with {}".format(
            model_args.llm_model_name_or_path
        )
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

    guide_message = (
        "\n"
        f"#############################################################################\n"
        f"##   A {model_name} chatbot is now chatting with you!\n"
        f"##   The command for loading a new image: ###Load image:\n"
        f"#############################################################################\n"
        "\n"
    )
    print(guide_message)

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )


    end_string = chatbot_args.end_string
    if chatbot_args.chatbot_type == "mini_gpt":
        context = "Give the following image: <Img>ImageContent</Img>. " + "You will be able to see the image once I provide it to you. Please answer my questions."
        user_name = "Human"
        sep = "###"

    elif chatbot_args.chatbot_type == "llava":
        context = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        user_name = "USER"
        sep = " "
    else:
        context = ""
        user_name = ""
        sep = "###"
    prompt_structure = chatbot_args.prompt_structure

    # Load image and input text for reasoning
    image_list = []
    if chatbot_args.image_path is not None:
        raw_image = Image.open(chatbot_args.image_path)
    else:
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    base_size = raw_image.size
    image_list.append(np.array(raw_image))
    input_text = chatbot_args.input_text
    if chatbot_args.task == "image_caption" and len(input_text) == 0:
        input_text = "a photography of"
    if chatbot_args.chatbot_type == "mini_gpt":
        context += sep + user_name + ": " + "<Img><ImageHere></Img> "
    elif chatbot_args.chatbot_type == "llava":
        context += sep + user_name + ": " + "<image>\n"

    # this flag is for determining if we need to add the ###Human: prompt
    # if text after loading image, we add it when loading image
    # else, we add it when read the text.
    text_after_loading_image = True

    if model_args.use_prompt_cache:
        if osp.exists(model_args.prompt_cache_path):
            model.backend_model.load_prompt_cache(
                model_args.prompt_cache_path)

    if chatbot_args.task == "image_caption":
        # single round reasoning
        input_dataset = dataset.from_dict({
            "type": "image_text",
            "instances": [{"images": np.stack(image_list),
                           "text":  input_text,}]
        })
        output = inferencer.inference(model, input_dataset)
        print(output.backend_dataset['text'])
    else:
        # text, 1st image token, answer, text, 2nd image token, 
        while True:
            input_text = input("User >>> ")
            if input_text == "exit":
                print("exit...")
                break
            elif input_text.startswith("###Load image:"):
                image_path = input_text[14:]
                try:
                    raw_image = Image.open(image_path)
                    # current dataset doesn't support batch of image with different shape
                    # so we resize the image and convert then into a numpy array
                    # In the future, we need to design a new dataset format that support 
                    # batch of image with different shape
                    raw_image = raw_image.resize(base_size)
                    image_list.append(np.array(raw_image))
                    if chatbot_args.chatbot_type == "mini_gpt":
                        context += sep + user_name + ": " + "<Img><ImageHere></Img> "
                    elif chatbot_args.chatbot_type == "llava":
                        context += sep + user_name + ": " + "<image>\n"
                    else:
                        raise NotImplementedError
                    text_after_loading_image = True
                    print("Finish loading image with path {}".format(image_path))
                    continue
                except FileNotFoundError:
                    print("Load image failed with path {}".format(image_path))
                    continue
            elif input_text == "reset":
                context = ""
                print("Chat history cleared")
                continue
            
            if text_after_loading_image is False:
                context += sep + user_name + ": "
            else:
                text_after_loading_image = False
            
            if not input_text:
                input_text = " "
            context += prompt_structure.format(input_text=input_text)

            # TODO handle when model doesn't have the get_max_length
            context = context[-model.get_max_length():]     # Memory of the bot
            input_dataset = dataset.from_dict({
                "type": "image_text",
                "instances": [{"images": np.stack(image_list),
                               "text":  context,}]
            })
            if chatbot_args.chatbot_type in ["mini_gpt", "llava"]:
                remove_image_flag = True
            else:
                remove_image_flag = False
            begin_time = time.time()
            if not chatbot_args.stream_inference:
                # directly inference the results
                output_dataset = inferencer.inference(
                    model,
                    input_dataset,
                    remove_image_flag=remove_image_flag,
                    chatbot_type=chatbot_args.chatbot_type,)
                response = output_dataset.backend_dataset['text']
                print(response[0])
                print("\n", end="")
                context += response[0]
            else:
                # do the stream inference
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
            
            if model_args.use_prompt_cache:
                osp.makedirs(osp.dirname(model_args.prompt_cache_path),
                             exist_ok=True)
                model.backend_model.save_prompt_cache(
                            model_args.prompt_cache_path)
            end_time = time.time()
            print("Whole data and model forward time", end_time - begin_time)
if __name__ == "__main__":
    main()
