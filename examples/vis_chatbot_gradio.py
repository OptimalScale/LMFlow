#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple Multimodal chatbot implemented with lmflow APIs.
"""
import logging
import json
import time

from PIL import Image
from lmflow.pipeline.inferencer import Inferencer

import numpy as np
import os
import sys
import torch
import warnings
import gradio as gr
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import (VisModelArguments, DatasetArguments, \
                         InferencerArguments, AutoArguments)

MAX_BOXES = 20

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")
torch.multiprocessing.set_start_method('spawn', force=True)

css = """
#user {
    float: right;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:	#9DC284;
    border-radius:5px;
    margin:10px 0px;
}

#chatbot {
    float: left;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:#7BA7D7;
    border-radius:5px;
    margin:10px 0px;
}
"""


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
        default="image_caption",
        metadata={
            "help": "task for reasoning",
        }
    )
    chatbot_format: Optional[str] = field(
        default="None",
        metadata={
            "help": "prompt format"
        }
    )


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state = ''
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(placeholder="Please upload an image first", interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list,
    )

def upload_image(image_file, history, text_input, chat_state, image_list):
    # if gr_image is None:
    #     return None, None, gr.update(interactive=True), chat_state, None
    history = history + [((image_file.name,), None)]

    if chat_state is None:
        if chatbot_args.chatbot_format == "mini_gpt":
            chat_state = "Give the following image: <Img>ImageContent</Img>. " + "You will be able to see the image once I provide it to you. Please answer my questions."
        else:
            chat_state = ''
    image = read_img(image_file.name)
    if not isinstance(image_list, list) or (
            isinstance(image_list, list) and len(image_list) == 0):
        image_list = []
        image_list.append(image)
    else:
        image_list.append(image.resize(image_list[0].size))

    if chatbot_args.chatbot_format == "mini_gpt":
        chat_state += "### Human: " + "<Img><ImageHere></Img>"
    return (
        gr.update(interactive=True, placeholder='Enter text and press enter, or upload an image'),
        history,
        chat_state,
        image_list,
    )

def read_img(image):
    if isinstance(image, str):
        raw_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        raw_image = image
    else:
        raise NotImplementedError
    return raw_image

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    prompted_user_message = prompt_structure.format(input_text=user_message)
    if chat_state is None:
        chat_state = ''
    chat_state = chat_state + prompted_user_message

    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, image_list, num_beams=1, temperature=1.0):
    input_dataset = dataset.from_dict({
        "type": "image_text",
        "instances": [{"images": np.stack([np.array(i) for i in image_list]),
                        "text": chat_state}]
    })
    remove_image_flag = chatbot_args.chatbot_format=="mini_gpt"

    chatbot[-1][1] = ''

    print_index = 0
    token_per_step = 4 # 48
    max_new_tokens = -1
    temperature = 0.7
    context = chatbot

    # Another user may have exited during the handling of his/her response,
    # Wait for inferencer process to complete its job, after which "busy" mark
    # in the request_queue will be released
    while not request_queue.empty():
        time.sleep(0.01)

    # Clean response_queue left by the previous user
    while not response_queue.empty():
        response_queue.get()

    request_queue.put((
        context,
        max_new_tokens,
        token_per_step,
        temperature,
        end_string,
        input_dataset,
        remove_image_flag
    ))

    while True:
        if not response_queue.empty():
            response, flag_break = response_queue.get()

            # Prints characters in the buffer
            new_print_index = print_index
            for char in response[print_index:]:
                if end_string is not None and char == end_string[0]:
                    if new_print_index + len(end_string) >= len(response):
                        break

                new_print_index += 1
                chatbot[-1][1] += char
                chat_state += char
                time.sleep(0.06)
                yield chatbot, chat_state, image_list

            print_index = new_print_index

            if flag_break:
                break

    char = "\n"
    chatbot[-1][1] += char
    chat_state += char
    yield chatbot, chat_state, image_list


def start_inferencer(
    request_queue,
    response_queue,
    model_args,
    pipeline_name,
    pipeline_args,
    data_args,
    dataset,
    chatbot_args,
):
    with open(pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
        custom_model=model_args.custom_model,
    )

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    while True:
        if not request_queue.empty():
            request_queue.put("busy")
            request = request_queue.get()

            context = request[0]
            max_new_tokens = request[1]
            token_per_step = request[2]
            temperature = request[3]
            end_string = request[4]
            input_dataset = request[5]
            remove_image_flag = request[6]

            break_in_the_middle = False
            for response_text, flag_break in inferencer.stream_inference(
                context=context,
                model=model,
                max_new_tokens=max_new_tokens,
                token_per_step=token_per_step,
                temperature=temperature,
                end_string=end_string,
                input_dataset=input_dataset,
                remove_image_flag=remove_image_flag,
            ):
                response_queue.put((response_text, flag_break))
                if flag_break:
                    break_in_the_middle = True
                    break

            if not break_in_the_middle:
                response_text = ''
                flag_break = True
                response_queue.put((response_text, flag_break))

            mark = ""
            while mark != "busy":
                mark = request_queue.get()     # Release the "busy" mark

        time.sleep(0.001)


if __name__ == "__main__":
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
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args, backend="dict")

    request_queue = torch.multiprocessing.Queue()
    response_queue = torch.multiprocessing.Queue()
    inferencer_process = torch.multiprocessing.Process(
        target=start_inferencer,
        args=(
            request_queue,
            response_queue,
            model_args,
            pipeline_name,
            pipeline_args,
            data_args,
            dataset,
            chatbot_args,
        ),
    )
    inferencer_process.start()

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure

    title = """<h1 align="center">LMFlow Multi-modal Chatbot</h1>"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)

        with gr.Row():
            chat_state = gr.State()
            image_list = gr.State()

            with gr.Column(scale=0.1, min_width=0):
                clear = gr.Button("Restart")

            with gr.Column(scale=0.8):
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Please upload an image first",
                    interactive=False,
                ).style(container=False)

            with gr.Column(scale=0.1, min_width=0):
                upload_button = gr.UploadButton("📁", file_types=["image"])

        txt_msg = text_input.submit(
            fn=gradio_ask,
            inputs=[text_input, chatbot, chat_state],
            outputs=[text_input, chatbot, chat_state],
            queue=False,
        ).then(
            fn=gradio_answer,
            inputs=[chatbot, chat_state, image_list],
            outputs=[chatbot, chat_state, image_list],
        )
        txt_msg.then(
            lambda: gr.update(interactive=True), None, [text_input], queue=False
        )

        file_msg = upload_button.upload(
            fn=upload_image,
            inputs=[upload_button, chatbot, text_input, chat_state, image_list],
            outputs=[text_input, chatbot, chat_state, image_list],
            queue=False,
        )

        clear.click(
            fn=gradio_reset,
            inputs=[chat_state, image_list],
            outputs=[chatbot, text_input, upload_button, chat_state, image_list],
            queue=False,
        )

    demo.queue(max_size=1, api_open=False).launch(share=True)
    inferencer_process.join()

