#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.

import logging
from logging import Logger
from typing import Dict, List, Union

import transformers
from transformers.testing_utils import CaptureLogger
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from lmflow.utils.conversation_template import ConversationTemplate
from lmflow.utils.constants import CONVERSATION_ROLE_NAMES
from lmflow.args import DatasetArguments


logger = logging.getLogger(__name__)
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


def blocking_paired(
    token_dict: Dict, 
    column_names: List,
    block_size: int, 
    model_max_length: int,
    pad_token_id: int,
    padding_side: str,
) -> Dict:
    block_size_warning_num = 0
    num_example = len(token_dict[list(token_dict.keys())[0]])
    for i in range(num_example):
        for column_name in column_names:
            max_length = min(block_size, model_max_length)
            pad_length = max_length - len(token_dict[f"input_ids_{column_name}"][i])
            if block_size < model_max_length:
                block_size_warning_num += 1
            if pad_length < 0:
                # Truncates too long samples
                for key in [f"input_ids_{column_name}", f"attention_mask_{column_name}"]:
                    token_dict[key][i] = token_dict[key][i][:pad_length]
            else:
                if padding_side == 'right':
                    # Pads too short samples
                    token_dict[f"input_ids_{column_name}"][i].extend(
                        [pad_token_id for _ in range(pad_length)]
                    )
                    token_dict[f"attention_mask_{column_name}"][i].extend(
                        [0 for _ in range(pad_length)]
                    )
                elif padding_side == 'left':
                    # Pads too short samples
                    token_dict[f"input_ids_{column_name}"][i] = (
                        [pad_token_id for _ in range(pad_length)] + token_dict[f"input_ids_{column_name}"][i]
                    )
                    token_dict[f"attention_mask_{column_name}"][i] = (
                        [0 for _ in range(pad_length)] + token_dict[f"attention_mask_{column_name}"][i]
                    )
                else:
                    raise ValueError(
                        f"padding_side should be either 'right' or 'left', got {padding_side}"
                    )
    if block_size_warning_num > 0:
        logger.warning(
            f"There are {block_size_warning_num} of {num_example} samples where"
            f" block_size {block_size} < model_max_length"
            f" {model_max_length}, use block_size"
            " for maximum tokenized sequence length"
        )
        
    return token_dict


def blocking(
    token_dict: Dict, 
    block_size: int, 
    model_max_length: int,
    pad_token_id: int,
    padding_side: str,
) -> Dict:
    block_size_warning_num = 0
    num_example = len(token_dict[list(token_dict.keys())[0]])
    for i in range(num_example):
        max_length = min(block_size, model_max_length)
        pad_length = max_length - len(token_dict["input_ids"][i])
        if block_size < model_max_length:
            block_size_warning_num += 1
        if pad_length < 0:
            # Truncates too long samples
            for key in ["input_ids", "attention_mask", "labels"]:
                token_dict[key][i] = token_dict[key][i][:pad_length]
        else:
            if padding_side == 'right':
                # Pads too short samples
                token_dict["input_ids"][i].extend(
                    [pad_token_id for _ in range(pad_length)]
                )
                token_dict["attention_mask"][i].extend(
                    [0 for _ in range(pad_length)]
                )
                token_dict["labels"][i].extend(
                    [-100 for _ in range(pad_length)]
                )
            elif padding_side == 'left':
                # Pads too short samples
                token_dict["input_ids"][i] = (
                    [pad_token_id for _ in range(pad_length)] + token_dict["input_ids"][i]
                )
                token_dict["attention_mask"][i] = (
                    [0 for _ in range(pad_length)] + token_dict["attention_mask"][i]
                )
                token_dict["labels"][i] = (
                    [-100 for _ in range(pad_length)] + token_dict["labels"][i]
                )
            else:
                raise ValueError(
                    f"padding_side should be either 'right' or 'left', got {padding_side}"
                )
    if block_size_warning_num > 0:
        logger.warning(
            f"There are {block_size_warning_num} of {num_example} samples where"
            f" block_size {block_size} < model_max_length"
            f" {model_max_length}, use block_size"
            " for maximum tokenized sequence length"
        )
        
    return token_dict


def blocking_text_to_textlist(
    token_dict: Dict, 
    block_size: int, 
    model_max_length: int,
    pad_token_id: int,
    padding_side: str,
) -> Dict:
    block_size_warning_num = 0
    num_example = len(token_dict[list(token_dict.keys())[0]])
    max_length = min(block_size, model_max_length)
    
    for example_idx in range(num_example):
        for content_idx in range(len(token_dict["input_ids"][example_idx])):
            pad_length = max_length - len(token_dict["input_ids"][example_idx][content_idx])
            if block_size < model_max_length:
                block_size_warning_num += 1
            if pad_length < 0:
                # Truncates too long samples
                token_dict["input_ids"][example_idx][content_idx] = token_dict["input_ids"][example_idx][content_idx][:pad_length]
            else:
                if padding_side == 'right':
                    # Pads too short samples
                    token_dict["input_ids"][example_idx][content_idx].extend(
                        [pad_token_id for _ in range(pad_length)]
                    )
                elif padding_side == 'left':
                    # Pads too short samples
                    token_dict["input_ids"][example_idx][content_idx] = (
                        [pad_token_id for _ in range(pad_length)] + token_dict["input_ids"][example_idx][content_idx]
                    )
                else:
                    raise ValueError(
                        f"padding_side should be either 'right' or 'left', got {padding_side}"
                    )
    if block_size_warning_num > 0:
        logger.warning(
            f"There are {block_size_warning_num} of {num_example} samples where"
            f" block_size {block_size} < model_max_length"
            f" {model_max_length}, use block_size"
            " for maximum tokenized sequence length"
        )
        
    return token_dict


def paired_conversation_tokenize_function(
    examples, 
    data_args: DatasetArguments,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
    column_names,
    conversation_template: ConversationTemplate,
) -> Dict:
    num_example = len(examples[column_names[0]])
    token_dict = {}
    for column_name in column_names:
        token_dict[f"input_ids_{column_name}"] = [[] for _ in range(num_example)]
        token_dict[f"attention_mask_{column_name}"] = [[] for _ in range(num_example)]
        
    with CaptureLogger(tok_logger) as cl:
        for i in range(num_example):
            for column_name in column_names:
                messages = examples[column_name][i]["messages"]
                system = examples[column_name][i].get("system", None)
                tools = examples[column_name][i].get("tools", None)
                if len(messages) < 2 or messages[0]['role'] != CONVERSATION_ROLE_NAMES['user']:
                    tok_logger.warning(
                        "Invalid instance encountered. Either the conversation has less than "
                        "one round or the first message is not from the user."
                    )
                    continue
            
                if len(messages) % 2 != 0:
                    logger.warning(
                        "The number of messages is not even, the last message will be ignored."
                    )
                    messages = messages[:-1]
                
                encoded_conversation = conversation_template.encode_conversation(
                    tokenizer=tokenizer,
                    messages=messages,
                    system=system,
                    tools=tools,
                )

                input_ids = []
                for turn_idx, (user_input, assistant_result) in enumerate(encoded_conversation):
                    input_ids += user_input + assistant_result
                    
                token_dict[f"input_ids_{column_name}"][i].extend(input_ids)
                token_dict[f"attention_mask_{column_name}"][i].extend([1] * len(input_ids))
                
    if data_args.disable_group_texts:
        token_dict = blocking_paired(
            token_dict=token_dict,
            column_names=column_names,
            block_size=data_args.block_size,
            model_max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
        )

    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return token_dict


def conversation_tokenize_function(
    examples, 
    data_args: DatasetArguments,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
    column_names,
    conversation_template: ConversationTemplate,
) -> Dict:
    """Handels conversation datasets tokenization
    """
    num_example = len(examples[column_names[0]])
    token_dict = {
        "input_ids": [[] for _ in range(num_example)],
        "attention_mask": [[] for _ in range(num_example)],
        "labels": [[] for _ in range(num_example)],
    }
    with CaptureLogger(tok_logger) as cl:
        for i in range(len(examples["messages"])):
            messages = examples["messages"][i]
            system = examples.get("system", [None] * num_example)[i]
            tools = examples.get("tools", [None] * num_example)[i]
            if len(messages) < 2 or messages[0]['role'] != CONVERSATION_ROLE_NAMES['user']:
                tok_logger.warning(
                    "Invalid instance encountered. Either the conversation has less than "
                    "one round or the first message is not from the user."
                )
                continue
        
            if len(messages) % 2 != 0:
                logger.warning(
                    "The number of messages is not even, the last message will be ignored."
                )
                messages = messages[:-1]
                
            encoded_conversation = conversation_template.encode_conversation(
                tokenizer=tokenizer,
                messages=messages,
                system=system,
                tools=tools,
            )

            input_ids, labels = [], []
            for turn_idx, (user_input, assistant_result) in enumerate(encoded_conversation):
                input_ids += user_input + assistant_result
                
                if data_args.train_on_prompt:
                    labels += user_input + assistant_result
                else:
                    labels += [-100] * len(user_input) + assistant_result
                
            token_dict["input_ids"][i].extend(input_ids)
            token_dict["attention_mask"][i].extend([1] * len(input_ids))
            token_dict["labels"][i].extend(labels)

    if data_args.disable_group_texts:
        token_dict = blocking(
            token_dict=token_dict,
            block_size=data_args.block_size,
            model_max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
        )

    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return token_dict

            
def tokenize_function(
    examples, 
    data_args: DatasetArguments,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    column_names,
    label_columns,
    tokenized_column_order,
    add_special_tokens,
    use_truncation,
) -> Dict:
    """Handels text_only and text2text datasets tokenization
    """
    num_example = len(examples[column_names[0]])
    token_dict = {
        "input_ids": [[] for _ in range(num_example)],
        "attention_mask": [[] for _ in range(num_example)],
        "labels": [[] for _ in range(num_example)],
    }
    with CaptureLogger(tok_logger) as cl:
        for column_name in tokenized_column_order:
            encoding = tokenizer(
                examples[column_name],
                add_special_tokens=add_special_tokens,
                truncation=use_truncation,
            )

            if column_name in label_columns:
                labels = encoding["input_ids"].copy()
            else:
                labels = [
                    [-100] * len(encoding["input_ids"][i])
                    for i in range(num_example)
                ]

            for i in range(num_example):
                token_dict["input_ids"][i].extend(
                    encoding["input_ids"][i]
                )
                token_dict["attention_mask"][i].extend(
                    encoding["attention_mask"][i]
                )
                token_dict["labels"][i].extend(labels[i])

    if data_args.disable_group_texts:
        token_dict = blocking(
            token_dict=token_dict,
            block_size=data_args.block_size,
            model_max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
        )

    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return token_dict


def text_to_textlist_tokenize_function(
    examples, 
    data_args: DatasetArguments,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    column_names,
    add_special_tokens,
    use_truncation,
) -> Dict:
    """For rm inference, and don't need attn mask and labels.
    NOTE: input_ids here refers to the tokenized input_ids of the input **and** output
    """
    num_example = len(examples[column_names[0]])
    output_dict = {column_name: examples[column_name] for column_name in column_names}
    output_dict["input_ids"] = [[] for _ in range(num_example)]
    
    for example_idx in range(num_example):
        encoded = tokenizer(
            [
                examples["input"][example_idx] + examples["output"][example_idx][i] 
                for i in range(len(examples["output"][example_idx]))
            ],
            add_special_tokens=add_special_tokens,
            truncation=use_truncation,
        )

        output_dict["input_ids"][example_idx] = encoded["input_ids"]
        
    if data_args.disable_group_texts:
        output_dict = blocking_text_to_textlist(
            token_dict=output_dict,
            block_size=data_args.block_size,
            model_max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
        )
        
    return output_dict