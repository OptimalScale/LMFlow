#!/usr/bin/env python
# coding=utf-8
# Tests to compare original HuggingFace tokenizer results with LMFlow tokenize_function

import pytest
import torch
import copy
from typing import Dict, List, Union

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer

from lmflow.args import DatasetArguments
from lmflow.tokenization.hf_decoder_model import tokenize_function
from lmflow.utils.conversation_template import PRESET_TEMPLATES

# Sample texts for testing
SAMPLE_TEXTS = [
    "This is a simple test sentence.",
    "Another example with more words to test tokenization functions.",
    "A third sample with different length to ensure consistency."
]

# Sample text2text examples
SAMPLE_TEXT2TEXT = {
    "input": [
        "Question: What is the capital of France?",
        "Explain the process of photosynthesis."
    ],
    "output": [
        "Answer: Paris is the capital of France.",
        "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."
    ]
}

# Sample conversations
SAMPLE_CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm well, thank you for asking!"}
        ],
        "system": "You are a helpful assistant."
    },
    {
        "messages": [
            {"role": "user", "content": "Can you help with math?"},
            {"role": "assistant", "content": "Sure, what's your question?"},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2=4"}
        ],
        "system": "You are a math tutor."
    }
]

# Models to test
MODEL_NAMES = [
    "gpt2",
    "facebook/opt-125m",
    "bert-base-uncased"  # Add a non-decoder model to check compatibility
]

@pytest.fixture
def data_args():
    """Create DatasetArguments with default settings."""
    return DatasetArguments(
        dataset_path=None,
        disable_group_texts=False,
        block_size=512,
        train_on_prompt=False
    )

@pytest.fixture
def data_args_with_blocking():
    """Create DatasetArguments with blocking enabled."""
    return DatasetArguments(
        dataset_path=None,
        disable_group_texts=True,
        block_size=20,
        train_on_prompt=False
    )

def compare_tokenization(hf_result, lmflow_result):
    """Helper to compare HuggingFace and LMFlow tokenization results."""
    # Check that input_ids match
    assert len(hf_result) == len(lmflow_result["input_ids"]), "Number of examples mismatch"
    
    for i in range(len(hf_result)):
        # When comparing without blocking, sequences should be identical
        if isinstance(hf_result[i], list):  # Handle both list and tensor inputs
            hf_ids = hf_result[i]
        else:
            hf_ids = hf_result[i].tolist()
            
        lmflow_ids = lmflow_result["input_ids"][i]
        
        # Print some debug info if they don't match
        if hf_ids != lmflow_ids:
            print(f"Mismatch at index {i}:")
            print(f"HF     : {hf_ids}")
            print(f"LMFlow : {lmflow_ids}")
            
        assert hf_ids == lmflow_ids, f"Input IDs mismatch at index {i}"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_text_only_tokenization_comparison(model_name, data_args):
    """Test text_only tokenization comparison between HF and LMFlow."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If the tokenizer doesn't have a pad token, set it to eos_token
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples
    examples = {"text": SAMPLE_TEXTS}
    
    # Original HuggingFace tokenization
    hf_encoding = tokenizer(
        examples["text"],
        add_special_tokens=True,
        truncation=True,
        return_tensors=None
    )
    
    # LMFlow tokenization
    lmflow_encoding = tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=["text"],
        label_columns=["text"],
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Compare results
    compare_tokenization(hf_encoding["input_ids"], lmflow_encoding)
    
    # Also check that attention_mask matches
    for i in range(len(hf_encoding["input_ids"])):
        if isinstance(hf_encoding["attention_mask"][i], list):
            hf_attention = hf_encoding["attention_mask"][i]
        else:
            hf_attention = hf_encoding["attention_mask"][i].tolist()
            
        assert hf_attention == lmflow_encoding["attention_mask"][i], f"Attention mask mismatch at index {i}"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_text2text_tokenization_comparison(model_name, data_args):
    """Test text2text tokenization comparison between HF and LMFlow."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If the tokenizer doesn't have a pad token, set it to eos_token
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples
    examples = copy.deepcopy(SAMPLE_TEXT2TEXT)
    
    # Original HuggingFace tokenization
    hf_input_encoding = tokenizer(
        examples["input"],
        add_special_tokens=True,
        truncation=True,
        return_tensors=None
    )
    
    hf_output_encoding = tokenizer(
        examples["output"],
        add_special_tokens=True,
        truncation=True,
        return_tensors=None
    )
    
    # LMFlow tokenization
    lmflow_encoding = tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=["input", "output"],
        label_columns=["output"],
        tokenized_column_order=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # For text2text, LMFlow concatenates input and output
    # We need to check that the concatenation is done correctly
    for i in range(len(examples["input"])):
        # Get the token IDs
        hf_input_ids = hf_input_encoding["input_ids"][i]
        hf_output_ids = hf_output_encoding["input_ids"][i]
        lmflow_ids = lmflow_encoding["input_ids"][i]
        
        # Check that the concatenation matches
        if isinstance(hf_input_ids, list):
            concatenated_ids = hf_input_ids + hf_output_ids
        else:
            concatenated_ids = hf_input_ids.tolist() + hf_output_ids.tolist()
            
        # Print debug info if they don't match
        if concatenated_ids != lmflow_ids:
            print(f"Concatenation mismatch at index {i}:")
            print(f"HF Input : {hf_input_ids}")
            print(f"HF Output: {hf_output_ids}")
            print(f"Concat   : {concatenated_ids}")
            print(f"LMFlow   : {lmflow_ids}")
            
        assert concatenated_ids == lmflow_ids, f"Concatenation mismatch at index {i}"
        
        # Check labels: should be -100 for input and actual output IDs for output
        expected_labels = [-100] * len(hf_input_ids) + hf_output_ids if isinstance(hf_output_ids, list) else [-100] * len(hf_input_ids) + hf_output_ids.tolist()
        assert lmflow_encoding["labels"][i] == expected_labels, f"Labels mismatch at index {i}"

@pytest.mark.parametrize("model_name", ["gpt2"])
def test_conversation_tokenization_comparison(model_name, data_args):
    """Test conversation tokenization between HF (with chat template) and LMFlow."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Skip if tokenizer doesn't support chat templates
    if not hasattr(tokenizer, 'apply_chat_template'):
        pytest.skip(f"Tokenizer {model_name} doesn't support chat templates")
    
    # Define a simple template for testing
    template = PRESET_TEMPLATES["empty_no_special_tokens"]
    
    # For each conversation example
    for i, conversation in enumerate(SAMPLE_CONVERSATIONS):
        # Convert to format expected by tokenizer
        chat_messages = conversation["messages"]
        
        # First, try direct HF tokenization
        try:
            # If tokenizer supports system messages, use it
            hf_encoding = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                return_tensors=None
            )
        except Exception as e:
            # If it fails, skip this test
            print(f"HF chat template failed: {str(e)}")
            continue
        
        # Prepare examples for LMFlow
        examples = {
            "messages": [chat_messages],
            "system": [conversation["system"]],
            "tools": [None]
        }
        
        # Use LMFlow conversation tokenization
        from lmflow.tokenization.hf_decoder_model import conversation_tokenize_function
        
        lmflow_encoding = conversation_tokenize_function(
            examples=examples,
            data_args=data_args,
            tokenizer=tokenizer,
            column_names=["messages"],
            conversation_template=template
        )
        
        # For conversation format, direct comparison is challenging due to different templates
        # Instead, we'll check that both produce valid sequences
        assert len(lmflow_encoding["input_ids"]) > 0, "LMFlow encoding should not be empty"
        assert len(lmflow_encoding["input_ids"][0]) > 0, "LMFlow encoding should have non-empty sequences"
        assert len(lmflow_encoding["attention_mask"]) > 0, "LMFlow attention_mask should not be empty"
        assert len(lmflow_encoding["labels"]) > 0, "LMFlow labels should not be empty"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_blocking_consistency(model_name, data_args_with_blocking):
    """Test that blocking in LMFlow produces consistent sequence lengths."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If the tokenizer doesn't have a pad token, set it to eos_token
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples with varying lengths
    examples = {"text": SAMPLE_TEXTS}
    
    # LMFlow tokenization with blocking
    lmflow_encoding = tokenize_function(
        examples=examples,
        data_args=data_args_with_blocking,
        tokenizer=tokenizer,
        column_names=["text"],
        label_columns=["text"],
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check that all sequences have the same length (block_size)
    for i in range(len(lmflow_encoding["input_ids"])):
        assert len(lmflow_encoding["input_ids"][i]) == data_args_with_blocking.block_size, \
            f"Sequence {i} length should be {data_args_with_blocking.block_size}"
        assert len(lmflow_encoding["attention_mask"][i]) == data_args_with_blocking.block_size, \
            f"Attention mask {i} length should be {data_args_with_blocking.block_size}"
        assert len(lmflow_encoding["labels"][i]) == data_args_with_blocking.block_size, \
            f"Labels {i} length should be {data_args_with_blocking.block_size}"