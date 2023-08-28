#!/usr/bin/env python
# coding=utf-8
# FIXME update the doc string.
"""This Python code defines a class Multi Modal Dataset.
"""
import copy
from dataclasses import dataclass, field
import json
from PIL import Image
import os.path as osp
import transformers
import torch
from torch.utils.data import Dataset

from lmflow.args import DatasetArguments
from lmflow.datasets import llava_conversation_lib as conversation_lib

from .llava_constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class CustomMultiModalDataset(Dataset):
    """Dataset for Multi Modal data"""

    def __init__(self, dataset_path: str,
                 data_args: DatasetArguments):
        super(CustomMultiModalDataset, self).__init__()
        data_dict = json.load(open(dataset_path, "r"))
        self.data_dict = data_dict
        print("Finish loading json file in dataset.")
        self.data_args = data_args
        self.image_folder = data_args.image_folder

    def __len__(self):
        return len(self.data_dict)

    def register_tokenizer(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = getattr(
            tokenizer, "image_processor", image_processor)

    def __getitem__(self, i):
        data = self.data_dict[i]
        if isinstance(i, int):
            data = [data]
        assert len(data) == 1
        processor = self.image_processor
        if 'image' in data[0]:
            image_file = data[0]['image']
            image = Image.open(
                osp.join(self.image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result    
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            data = preprocess_multimodal_llava(
                copy.deepcopy([e["conversations"] for e in data]),
                self.data_args)
        else:
            data = copy.deepcopy([e["conversations"] for e in data])
        data_dict = preprocess_llama_from_llava(
            data,
            self.tokenizer,
            has_image=('image' in self.data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        # image exist in the data
        if 'image' in self.data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.crop_size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
        return data_dict



def preprocess_multimodal_llava(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                                DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.use_image_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + \
                    replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def tokenizer_image_token(prompt,
                          tokenizer,
                          image_token_index=IMAGE_TOKEN_INDEX,
                          return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_llama_from_llava(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False):
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        return batch
