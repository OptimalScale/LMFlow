#!/usr/bin/env python
# coding=utf-8
# Tests for HF text regression model tokenization functions

'''
Tests for text regression model tokenization, including paired conversations and text-to-textlist functions.
'''

import pytest
import torch
import copy
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.testing_utils import CaptureLogger

from lmflow.args import DatasetArguments
from lmflow.utils.conversation_template import ConversationTemplate, PRESET_TEMPLATES
from lmflow.tokenization.hf_text_regression_model import (
    blocking_paired,
    blocking,
    blocking_text_to_textlist,
    paired_conversation_tokenize_function,
    conversation_tokenize_function,
    tokenize_function,
    text_to_textlist_tokenize_function
)
from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

# Sample text for testing
SAMPLE_TEXT = "This is a test input for tokenization functions."
SAMPLE_TEXT_ALT = "This is an alternative text for testing."

# Sample conversation for testing
SAMPLE_CONVERSATION = {
    "system": "You are a helpful assistant.",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking!"
        },
        {
            "role": "user",
            "content": "Can you help me with a task?"
        },
        {
            "role": "assistant",
            "content": "Of course! I'd be happy to help. What do you need assistance with?"
        }
    ]
}

# Paired conversations for testing
SAMPLE_PAIRED_CONVERSATION = {
    "col1": [
        {
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
            ]
        },
        {
            "system": "You are a math tutor.",
            "messages": [
                {"role": "user", "content": "Can you solve 2+2?"},
                {"role": "assistant", "content": "2+2=4"}
            ]
        }
    ],
    "col2": [
        {
            "system": "You are a science teacher.",
            "messages": [
                {"role": "user", "content": "What is gravity?"},
                {"role": "assistant", "content": "Gravity is a force that attracts objects to each other."}
            ]
        },
        {
            "system": "You are a history expert.",
            "messages": [
                {"role": "user", "content": "Who was Napoleon?"},
                {"role": "assistant", "content": "Napoleon Bonaparte was a French military leader and emperor."}
            ]
        }
    ]
}

# Sample for text_to_textlist testing
SAMPLE_TEXT_TO_TEXTLIST = {
    "input": [
        "Rate these responses from 1-5:",
        "Choose the best answer:"
    ],
    "output": [
        ["Response A is good.", "Response B is better.", "Response C is best."],
        ["Option X", "Option Y", "Option Z"]
    ]
}

# Fixtures for common test objects
@pytest.fixture
def data_args():
    return DatasetArguments(
        dataset_path=None,
        disable_group_texts=False,
        block_size=512,
        train_on_prompt=False
    )

@pytest.fixture
def data_args_with_blocking():
    return DatasetArguments(
        dataset_path=None,
        disable_group_texts=True,
        block_size=16,
        train_on_prompt=False
    )

@pytest.fixture
def data_args_with_train_on_prompt():
    return DatasetArguments(
        dataset_path=None,
        disable_group_texts=False,
        block_size=512,
        train_on_prompt=True
    )

@pytest.fixture
def gpt2_tokenizer():
    # Use a real tokenizer for more realistic tests
    return AutoTokenizer.from_pretrained('gpt2')

# Test blocking_paired function with real tokenizer
def test_blocking_paired_real_data(gpt2_tokenizer):
    # Encode sample texts
    encoded_1 = gpt2_tokenizer(
        [SAMPLE_TEXT, "Short text"],
        return_tensors=None,
        padding=False,
        truncation=False
    )
    
    encoded_2 = gpt2_tokenizer(
        [SAMPLE_TEXT_ALT, "Another short text"],
        return_tensors=None,
        padding=False,
        truncation=False
    )
    
    # Create token dict for paired data
    token_dict = {
        "input_ids_col1": encoded_1["input_ids"],
        "attention_mask_col1": encoded_1["attention_mask"],
        "input_ids_col2": encoded_2["input_ids"],
        "attention_mask_col2": encoded_2["attention_mask"]
    }
    
    # Apply blocking_paired
    result = blocking_paired(
        token_dict=token_dict,
        column_names=["col1", "col2"],
        block_size=20,
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,
        padding_side="right"
    )
    
    # Check that each sequence is exactly block_size
    assert len(result["input_ids_col1"][0]) == 20
    assert len(result["attention_mask_col1"][0]) == 20
    assert len(result["input_ids_col2"][0]) == 20
    assert len(result["attention_mask_col2"][0]) == 20
    
    # Check that original tokens are preserved
    assert result["input_ids_col1"][0][:len(encoded_1["input_ids"][0])] == encoded_1["input_ids"][0]
    assert result["input_ids_col2"][0][:len(encoded_2["input_ids"][0])] == encoded_2["input_ids"][0]
    
    # Test with left padding
    result_left_pad = blocking_paired(
        token_dict=copy.deepcopy(token_dict),
        column_names=["col1", "col2"],
        block_size=20,
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,
        padding_side="left"
    )
    
    # Check left padding was applied
    pad_length1 = 20 - len(encoded_1["input_ids"][0])
    pad_length2 = 20 - len(encoded_2["input_ids"][0])
    
    assert result_left_pad["input_ids_col1"][0][:pad_length1] == [0] * pad_length1
    assert result_left_pad["input_ids_col2"][0][:pad_length2] == [0] * pad_length2

# Test blocking_text_to_textlist function with real tokenizer
def test_blocking_text_to_textlist_real_data(gpt2_tokenizer):
    # Create sample input_ids for text_to_textlist structure
    input_prompts = ["Rank these options:", "Choose the best:"]
    options = [
        ["Option A", "Option B", "Option C"],
        ["Choice X", "Choice Y"]
    ]
    
    # Tokenize all combinations of prompt + option
    input_ids = []
    for i, prompt in enumerate(input_prompts):
        input_ids.append([])
        for option in options[i]:
            combined = prompt + " " + option
            encoded = gpt2_tokenizer(combined)["input_ids"]
            input_ids[i].append(encoded)
    
    # Create token dict
    token_dict = {
        "input": input_prompts,
        "output": options,
        "input_ids": input_ids
    }
    
    # Apply blocking_text_to_textlist
    result = blocking_text_to_textlist(
        token_dict=token_dict,
        block_size=15,
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,
        padding_side="right"
    )
    
    # Check that each sequence is exactly block_size
    for example_idx in range(len(result["input_ids"])):
        for option_idx in range(len(result["input_ids"][example_idx])):
            assert len(result["input_ids"][example_idx][option_idx]) == 15
    
    # Check original tokens are preserved
    original_length = len(token_dict["input_ids"][0][0])
    if original_length <= 15:  # If no truncation happened
        assert result["input_ids"][0][0][:original_length] == token_dict["input_ids"][0][0]
    else:  # If truncation happened
        assert result["input_ids"][0][0] == token_dict["input_ids"][0][0][:15]
    
    # Test with left padding
    result_left_pad = blocking_text_to_textlist(
        token_dict=copy.deepcopy(token_dict),
        block_size=15,
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,
        padding_side="left"
    )
    
    # Check left padding was applied if original is shorter than block_size
    if original_length < 15:
        pad_length = 15 - original_length
        assert result_left_pad["input_ids"][0][0][:pad_length] == [0] * pad_length

# Test paired_conversation_tokenize_function with real template and tokenizer
def test_paired_conversation_tokenize_function_real(gpt2_tokenizer, data_args, data_args_with_blocking):
    # Create conversation template
    conversation_template = PRESET_TEMPLATES["empty_no_special_tokens"]
    
    # Call paired_conversation_tokenize_function
    result = paired_conversation_tokenize_function(
        examples=SAMPLE_PAIRED_CONVERSATION,
        data_args=data_args,
        tokenizer=gpt2_tokenizer,
        column_names=["col1", "col2"],
        conversation_template=conversation_template
    )
    
    # Check structure
    assert "input_ids_col1" in result
    assert "attention_mask_col1" in result
    assert "input_ids_col2" in result
    assert "attention_mask_col2" in result
    
    # Verify non-empty sequences
    assert len(result["input_ids_col1"][0]) > 0
    assert len(result["input_ids_col2"][0]) > 0
    
    # Test with blocking enabled
    result_blocked = paired_conversation_tokenize_function(
        examples=SAMPLE_PAIRED_CONVERSATION,
        data_args=data_args_with_blocking,
        tokenizer=gpt2_tokenizer,
        column_names=["col1", "col2"],
        conversation_template=conversation_template
    )
    
    # Check blocking was applied
    assert len(result_blocked["input_ids_col1"][0]) == data_args_with_blocking.block_size
    assert len(result_blocked["input_ids_col2"][0]) == data_args_with_blocking.block_size

# Test the conversation_tokenize_function in text_regression_model
def test_text_regression_conversation_tokenize_function(gpt2_tokenizer, data_args, data_args_with_train_on_prompt):
    # Create conversation examples
    examples = {
        "messages": [SAMPLE_CONVERSATION["messages"]],
        "system": [SAMPLE_CONVERSATION["system"]],
        "tools": [None]
    }
    
    # Use a simple conversation template
    conversation_template = PRESET_TEMPLATES["empty_no_special_tokens"]
    
    # Call conversation_tokenize_function
    result = conversation_tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=gpt2_tokenizer,
        column_names=["messages"],
        conversation_template=conversation_template
    )
    
    # Check structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Verify non-empty sequences
    assert len(result["input_ids"][0]) > 0
    assert len(result["attention_mask"][0]) > 0
    
    # Test with train_on_prompt option
    result_with_prompt = conversation_tokenize_function(
        examples=examples,
        data_args=data_args_with_train_on_prompt,
        tokenizer=gpt2_tokenizer,
        column_names=["messages"],
        conversation_template=conversation_template
    )
    
    # With train_on_prompt, we should have more non-100 labels
    non_label_count = sum(1 for label in result["labels"][0] if label != -100)
    non_label_count_with_prompt = sum(1 for label in result_with_prompt["labels"][0] if label != -100)
    
    assert non_label_count_with_prompt >= non_label_count

# Test text_to_textlist_tokenize_function with real tokenizer
def test_text_to_textlist_tokenize_function_real(gpt2_tokenizer, data_args, data_args_with_blocking):
    # Call text_to_textlist_tokenize_function
    result = text_to_textlist_tokenize_function(
        examples=SAMPLE_TEXT_TO_TEXTLIST,
        data_args=data_args,
        tokenizer=gpt2_tokenizer,
        column_names=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check structure - should maintain input, output and add input_ids
    assert "input" in result
    assert "output" in result
    assert "input_ids" in result
    
    # Check that input_ids has correct structure:
    # [example_idx][option_idx] -> token list
    assert len(result["input_ids"]) == len(SAMPLE_TEXT_TO_TEXTLIST["input"])
    for example_idx in range(len(result["input_ids"])):
        assert len(result["input_ids"][example_idx]) == len(SAMPLE_TEXT_TO_TEXTLIST["output"][example_idx])
    
    # Test with blocking enabled
    result_blocked = text_to_textlist_tokenize_function(
        examples=SAMPLE_TEXT_TO_TEXTLIST,
        data_args=data_args_with_blocking,
        tokenizer=gpt2_tokenizer,
        column_names=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check blocking was applied
    for example_idx in range(len(result_blocked["input_ids"])):
        for option_idx in range(len(result_blocked["input_ids"][example_idx])):
            assert len(result_blocked["input_ids"][example_idx][option_idx]) == data_args_with_blocking.block_size

# Test tokenize_function in text_regression_model with real tokenizer
def test_text_regression_tokenize_function(gpt2_tokenizer, data_args):
    # Create example data
    examples = {
        "text": [SAMPLE_TEXT, SAMPLE_TEXT_ALT],
        "input": ["Question: What is NLP?", "Question: What is ML?"],
        "output": ["Natural Language Processing...", "Machine Learning..."]
    }
    
    # Call tokenize_function for text_only data
    result = tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=gpt2_tokenizer,
        column_names=["text"],
        label_columns=["text"],  # text is both input and label
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # For text_only, labels should match input_ids
    assert result["input_ids"][0] == result["labels"][0]
    
    # Test text2text
    result_text2text = tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=gpt2_tokenizer,
        column_names=["input", "output"],
        label_columns=["output"],  # Only output is label
        tokenized_column_order=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Verify input + output structure
    input_tokens = gpt2_tokenizer(examples["input"][0])["input_ids"]
    input_length = len(input_tokens)
    
    # The first input_length labels should be -100
    assert all(label == -100 for label in result_text2text["labels"][0][:input_length])
    
    # Test with blocking enabled
    result_blocked = tokenize_function(
        examples=examples,
        data_args=data_args_with_blocking,
        tokenizer=gpt2_tokenizer,
        column_names=["text"],
        label_columns=["text"],
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check that blocking was applied
    assert len(result_blocked["input_ids"][0]) == data_args_with_blocking.block_size