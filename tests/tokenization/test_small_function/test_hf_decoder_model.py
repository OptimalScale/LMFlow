#!/usr/bin/env python
# coding=utf-8
# Tests for HF decoder model tokenization functions

'''
Tests specific to the HF decoder model tokenization functions with realistic data.
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
from lmflow.tokenization.hf_decoder_model import (
    blocking,
    tokenize_function,
    conversation_tokenize_function
)
from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

# Sample text for testing
SAMPLE_TEXT = "This is a test input for tokenization functions."

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

# Test blocking function with real tokenizer and data
def test_blocking_real_data(gpt2_tokenizer):
    # Encode sample text
    encoded = gpt2_tokenizer(
        [SAMPLE_TEXT, "Short text"],
        return_tensors=None,
        padding=False,
        truncation=False
    )
    
    # Create token dict similar to what would be produced during tokenization
    token_dict = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": copy.deepcopy(encoded["input_ids"])
    }
    
    # Apply blocking with right padding
    result = blocking(
        token_dict=token_dict,
        block_size=20,  # Fixed block size
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,  # gpt2 has no pad token
        padding_side="right"
    )
    
    # Check that each sequence is exactly block_size
    for seq in result["input_ids"]:
        assert len(seq) == 20
    
    # Check that original tokens are preserved
    original_length = len(token_dict["input_ids"][0])
    assert result["input_ids"][0][:original_length] == token_dict["input_ids"][0]
    
    # Test blocking with left padding
    result_left_pad = blocking(
        token_dict=copy.deepcopy(token_dict),
        block_size=20,
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,
        padding_side="left"
    )
    
    # Check that padding is applied on the left
    pad_length = 20 - len(token_dict["input_ids"][0])
    assert result_left_pad["input_ids"][0][:pad_length] == [0] * pad_length
    
    # Test blocking with truncation
    result_truncated = blocking(
        token_dict=copy.deepcopy(token_dict),
        block_size=5,  # Smaller than the sequence
        model_max_length=gpt2_tokenizer.model_max_length,
        pad_token_id=gpt2_tokenizer.pad_token_id or 0,
        padding_side="right"
    )
    
    # Check that sequences are truncated to block_size
    assert len(result_truncated["input_ids"][0]) == 5
    assert result_truncated["input_ids"][0] == token_dict["input_ids"][0][:5]

# Test tokenize_function with real tokenizer and data
def test_tokenize_function_text_only(gpt2_tokenizer, data_args, data_args_with_blocking):
    # Create example data
    examples = {"text": [SAMPLE_TEXT, "Another example text."]}
    
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
    
    # Check with blocking enabled
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

def test_tokenize_function_text2text(gpt2_tokenizer, data_args):
    # Create example data
    examples = {
        "input": ["Question: What is tokenization?", "Query: Explain NLP."],
        "output": ["Tokenization is the process...", "Natural Language Processing is..."]
    }
    
    # Call tokenize_function for text2text data
    result = tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=gpt2_tokenizer,
        column_names=["input", "output"],
        label_columns=["output"],  # Only output is label
        tokenized_column_order=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # For text2text, labels should be -100 for input tokens and match output tokens for output
    input_tokens = gpt2_tokenizer(examples["input"][0])["input_ids"]
    input_length = len(input_tokens)
    
    # The first input_length labels should be -100
    assert all(label == -100 for label in result["labels"][0][:input_length])
    
    # The remaining labels should match the encoded output
    output_tokens = gpt2_tokenizer(examples["output"][0])["input_ids"]
    assert result["labels"][0][input_length:input_length + len(output_tokens)] == output_tokens

# Test conversation_tokenize_function with jinja template
def test_conversation_tokenize_function_jinja(gpt2_tokenizer, data_args, data_args_with_train_on_prompt):
    # Create conversation examples
    examples = {
        "messages": [SAMPLE_CONVERSATION["messages"]],
        "system": [SAMPLE_CONVERSATION["system"]],
        "tools": [None]
    }
    
    # Simple jinja template for testing
    test_template = "{% for message in conversation %}{% if message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% endfor %}"
    
    # Call conversation_tokenize_function with jinja template
    with patch.object(gpt2_tokenizer, "apply_chat_template") as mock_apply_template:
        # Mock the return value of apply_chat_template
        mock_apply_template.return_value = {
            "input_ids": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "assistant_masks": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Second half is assistant
        }
        
        result = conversation_tokenize_function(
            examples=examples,
            data_args=data_args,
            tokenizer=gpt2_tokenizer,
            column_names=["messages"],
            conversation_template=test_template
        )
    
    # Check structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Check that tokenizer.apply_chat_template was called
    mock_apply_template.assert_called_once()
    
    # Check labels: -100 for non-assistant tokens, actual token ids for assistant tokens
    assert result["labels"][0][:5] == [-100] * 5  # First 5 tokens (non-assistant)
    assert result["labels"][0][5:] == [55, 56, 57, 58, 59]  # Last 5 tokens (assistant)
    
    # Test with train_on_prompt=True
    with patch.object(gpt2_tokenizer, "apply_chat_template") as mock_apply_template:
        mock_apply_template.return_value = {
            "input_ids": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "assistant_masks": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        }
        
        result_train_on_prompt = conversation_tokenize_function(
            examples=examples,
            data_args=data_args_with_train_on_prompt,
            tokenizer=gpt2_tokenizer,
            column_names=["messages"],
            conversation_template=test_template
        )
    
    # With train_on_prompt, all tokens should be included in labels
    assert result_train_on_prompt["labels"][0] == [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

# Test conversation_tokenize_function with ConversationTemplate object
def test_conversation_tokenize_function_template_obj(gpt2_tokenizer, data_args, data_args_with_blocking):
    # Create conversation examples
    examples = {
        "messages": [SAMPLE_CONVERSATION["messages"]],
        "system": [SAMPLE_CONVERSATION["system"]],
        "tools": [None]
    }
    
    # Use a simple conversation template (e.g., empty_no_special_tokens)
    conversation_template = PRESET_TEMPLATES["empty_no_special_tokens"]
    
    # Call conversation_tokenize_function with template object
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
    
    # Verify that each turn is properly handled
    # We should have input_ids from all turns
    total_length = sum(len(msg["content"]) for msg in SAMPLE_CONVERSATION["messages"])
    assert len(result["input_ids"][0]) > 0
    
    # Test with blocking enabled
    result_blocked = conversation_tokenize_function(
        examples=examples,
        data_args=data_args_with_blocking,
        tokenizer=gpt2_tokenizer,
        column_names=["messages"],
        conversation_template=conversation_template
    )
    
    # Check that blocking was applied
    assert len(result_blocked["input_ids"][0]) == data_args_with_blocking.block_size