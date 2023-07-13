#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple Multimodal chatbot implemented with lmflow APIs.
"""
from dataclasses import dataclass, field
import logging
import json
from PIL import Image
from lmflow.pipeline.inferencer import Inferencer

import numpy as np
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
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

title = """
<h1 align="center">LMFlow-CHAT</h1>
<link rel="stylesheet" href="/path/to/styles/default.min.css">
<script src="/path/to/highlight.min.js"></script>
<script>hljs.highlightAll();</script>

<img src="https://optimalscale.github.io/LMFlow/_static/logo.png" alt="LMFlow" style="width: 30%; min-width: 60px; display: block; margin: auto; background-color: transparent;">

<p>LMFlow is in extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community.</p>

<p>We have thoroughly tested this toolkit and are pleased to make it available under <a class="reference external" href="https://github.com/OptimalScale/LMFlow">Github</a>.</p>
"""
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
    prompt_format: Optional[str] = field(
        default="None",
        metadata={
            "help": "prompt format"
        }
    )

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

with open(pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(
    model_args,
    tune_strategy='none',
    ds_config=ds_config,
    device=pipeline_args.device,
    custom_model=model_args.custom_model,
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


end_string = chatbot_args.end_string
prompt_structure = chatbot_args.prompt_structure


token_per_step = 4



title = """<h1 align="center">Demo of Multi-modality chatbot from LMFlow</h1>"""
description = """<h3>This is the demo of Multi-modality chatbot from LMFlow. Upload your images and start chatting!</h3>"""
# article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
# """

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state = ''
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(placeholder='Please upload your image first', interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list,
    )

def upload_image(image_file, history, text_input, chat_state, image_list):
    # if gr_image is None:
    #     return None, None, gr.update(interactive=True), chat_state, None
    history = history + [((image_file.name,), None)]

    if chat_state is None:
        if chatbot_args.prompt_format == "mini_gpt":
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

    if chatbot_args.prompt_format == "mini_gpt":
        chat_state += "Human: " + "<Img><ImageHere></Img>"
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
    remove_image_flag = chatbot_args.prompt_format=="mini_gpt"

    output_dataset = inferencer.inference(model, input_dataset, 
                        remove_image_flag=remove_image_flag)
    response = output_dataset.backend_dataset['text']
    chatbot[-1][-1] = response[0]
    chat_state += response[0]
    return chatbot, chat_state, image_list


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)

    with gr.Row():
        chat_state = gr.State()
        image_list = gr.State()

        with gr.Column(scale=0.1, min_width=0):
            clear = gr.Button("Restart")

        with gr.Column(scale=0.8):
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
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


demo.launch(share=True, enable_queue=True)
