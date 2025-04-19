#!/usr/bin/env python
# coding=utf-8
# Tests for tokenization functions

'''
Tests for core tokenization functions like blocking, tokenize_function, and conversation_tokenize_function
'''

import pytest
import torch
import copy
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import CaptureLogger

from lmflow.args import DatasetArguments
from lmflow.utils.conversation_template import ConversationTemplate
from lmflow.tokenization.hf_decoder_model import (
    blocking,
    tokenize_function,
    conversation_tokenize_function
)
from lmflow.tokenization.hf_text_regression_model import (
    blocking_paired,
    blocking_text_to_textlist,
    paired_conversation_tokenize_function,
    conversation_tokenize_function as text_regression_conversation_tokenize_function,
    tokenize_function as text_regression_tokenize_function,
    text_to_textlist_tokenize_function
)
from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

########################################################################
############################# Test data ################################
############################# Test data ################################
############################# Test data ################################
########################################################################
@pytest.fixture
def sample_token_dict():
    return {
        "input_ids": [[1, 2, 3, 4], [5, 6, 7]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 1]],
        "labels": [[1, 2, 3, 4], [5, 6, 7]]
    }

@pytest.fixture
def sample_token_dict_paired():
    return {
        "input_ids_col1": [[1, 2, 3, 4], [5, 6, 7]],
        "attention_mask_col1": [[1, 1, 1, 1], [1, 1, 1]],
        "input_ids_col2": [[8, 9, 10], [11, 12, 13, 14]],
        "attention_mask_col2": [[1, 1, 1], [1, 1, 1, 1]]
    }

@pytest.fixture
def sample_text_to_textlist_dict():
    return {
        "input": ["input1", "input2"],
        "output": [["output1_1", "output1_2"], ["output2_1", "output2_2"]],
        "input_ids": [
            [[1, 2, 3], [4, 5, 6]], 
            [[7, 8], [9, 10, 11]]
        ]
    }

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.model_max_length = 10
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])
    tokenizer.return_value = {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}
    return tokenizer

@pytest.fixture
def mock_conversation_template():
    template = MagicMock(spec=ConversationTemplate)
    template.encode_conversation = MagicMock(
        return_value=[([1, 2, 3], [4, 5])]
    )
    return template

@pytest.fixture
def sample_examples():
    return {
        "text": ["This is a test.", "Another example."],
        "input": ["Question 1", "Question 2"],
        "output": ["Answer 1", "Answer 2"],
        "messages": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good, thanks!"}
            ]
        ],
        "system": ["System 1", "System 2"]
    }

@pytest.fixture
def sample_conversation_examples():
    return {
        "messages": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good, thanks!"}
            ]
        ],
        "system": ["System 1", "System 2"],
        "tools": [None, None]
    }

@pytest.fixture
def sample_paired_conversation_examples():
    return {
        "col1": [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ],
                "system": "System 1"
            },
            {
                "messages": [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "Good, thanks!"}
                ],
                "system": "System 2"
            }
        ],
        "col2": [
            {
                "messages": [
                    {"role": "user", "content": "Tell me a joke"},
                    {"role": "assistant", "content": "Why did the chicken..."}
                ],
                "system": "System 1"
            },
            {
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "It's sunny!"}
                ],
                "system": "System 2"
            }
        ]
    }

@pytest.fixture
def data_args():
    args = DatasetArguments(
        dataset_path=None,
        disable_group_texts=False,
        block_size=8,
        train_on_prompt=False
    )
    return args

@pytest.fixture
def data_args_with_train_on_prompt():
    args = DatasetArguments(
        dataset_path=None,
        disable_group_texts=False,
        block_size=8,
        train_on_prompt=True
    )
    return args




#######################################################################
################### Tests for blocking function #######################
################### Tests for blocking function #######################
################### Tests for blocking function #######################
################### Tests for blocking function #######################
################### Tests for blocking function #######################
#######################################################################
def test_blocking_no_padding_no_truncation(sample_token_dict):
    token_dict = copy.deepcopy(sample_token_dict)
    result = blocking(
        token_dict=token_dict,
        block_size=10,
        model_max_length=10,
        pad_token_id=0,
        padding_side="right"
    )
    
    # Lengths should be padded to match block_size
    assert len(result["input_ids"][0]) == 10
    assert len(result["attention_mask"][0]) == 10
    assert len(result["labels"][0]) == 10
    
    # First 4 tokens should be unchanged for the first example
    assert result["input_ids"][0][:4] == [1, 2, 3, 4]
    assert result["attention_mask"][0][:4] == [1, 1, 1, 1]
    assert result["labels"][0][:4] == [1, 2, 3, 4]
    
    # Padding should be applied
    assert result["input_ids"][0][4:] == [0, 0, 0, 0, 0, 0]
    assert result["attention_mask"][0][4:] == [0, 0, 0, 0, 0, 0]
    assert result["labels"][0][4:] == [-100, -100, -100, -100, -100, -100]

def test_blocking_with_truncation(sample_token_dict):
    token_dict = copy.deepcopy(sample_token_dict)
    result = blocking(
        token_dict=token_dict,
        block_size=3,  # Smaller than first sequence
        model_max_length=10,
        pad_token_id=0,
        padding_side="right"
    )
    
    # Should truncate to block_size
    assert len(result["input_ids"][0]) == 3
    assert len(result["attention_mask"][0]) == 3
    assert len(result["labels"][0]) == 3
    
    # Should contain first 3 tokens only
    assert result["input_ids"][0] == [1, 2, 3]
    assert result["attention_mask"][0] == [1, 1, 1]
    assert result["labels"][0] == [1, 2, 3]

def test_blocking_left_padding(sample_token_dict):
    token_dict = copy.deepcopy(sample_token_dict)
    result = blocking(
        token_dict=token_dict,
        block_size=10,
        model_max_length=10,
        pad_token_id=0,
        padding_side="left"
    )
    
    # First example should be padded on the left
    assert result["input_ids"][0][:6] == [0, 0, 0, 0, 0, 0]
    assert result["attention_mask"][0][:6] == [0, 0, 0, 0, 0, 0]
    assert result["labels"][0][:6] == [-100, -100, -100, -100, -100, -100]
    
    # Original tokens should be at the end
    assert result["input_ids"][0][6:] == [1, 2, 3, 4]
    assert result["attention_mask"][0][6:] == [1, 1, 1, 1]
    assert result["labels"][0][6:] == [1, 2, 3, 4]

def test_blocking_left_truncation(sample_token_dict):
    token_dict = copy.deepcopy(sample_token_dict)
    result = blocking(
        token_dict=token_dict,
        block_size=3,
        model_max_length=10,
        pad_token_id=0,
        padding_side="right",
        truncation_side="left"
    )
    
    # Should truncate from the left
    assert result["input_ids"][0] == [2, 3, 4]
    assert result["attention_mask"][0] == [1, 1, 1]
    assert result["labels"][0] == [2, 3, 4]

def test_blocking_invalid_truncation_side(sample_token_dict):
    token_dict = copy.deepcopy(sample_token_dict)
    with pytest.raises(ValueError, match="truncation_side should be either 'right' or 'left'"):
        blocking(
            token_dict=token_dict,
            block_size=3,
            model_max_length=10,
            pad_token_id=0,
            padding_side="right",
            truncation_side="invalid"
        )

def test_blocking_invalid_padding_side(sample_token_dict):
    token_dict = copy.deepcopy(sample_token_dict)
    with pytest.raises(ValueError, match="padding_side should be either 'right' or 'left'"):
        blocking(
            token_dict=token_dict,
            block_size=10,
            model_max_length=10,
            pad_token_id=0,
            padding_side="invalid"
        )

# Tests for blocking_paired function
def test_blocking_paired(sample_token_dict_paired):
    token_dict = copy.deepcopy(sample_token_dict_paired)
    result = blocking_paired(
        token_dict=token_dict,
        column_names=["col1", "col2"],
        block_size=5,
        model_max_length=10,
        pad_token_id=0,
        padding_side="right"
    )
    
    # Check lengths
    assert len(result["input_ids_col1"][0]) == 5
    assert len(result["attention_mask_col1"][0]) == 5
    assert len(result["input_ids_col2"][0]) == 5
    assert len(result["attention_mask_col2"][0]) == 5
    
    # Check padding for first column, first example
    assert result["input_ids_col1"][0] == [1, 2, 3, 4, 0]
    assert result["attention_mask_col1"][0] == [1, 1, 1, 1, 0]
    
    # Check padding for second column, first example
    assert result["input_ids_col2"][0] == [8, 9, 10, 0, 0]
    assert result["attention_mask_col2"][0] == [1, 1, 1, 0, 0]

def test_blocking_paired_truncation(sample_token_dict_paired):
    token_dict = copy.deepcopy(sample_token_dict_paired)
    result = blocking_paired(
        token_dict=token_dict,
        column_names=["col1", "col2"],
        block_size=2,  # Smaller than all sequences
        model_max_length=10,
        pad_token_id=0,
        padding_side="right"
    )
    
    # Check truncation
    assert result["input_ids_col1"][0] == [1, 2]
    assert result["attention_mask_col1"][0] == [1, 1]
    assert result["input_ids_col2"][0] == [8, 9]
    assert result["attention_mask_col2"][0] == [1, 1]

# Tests for blocking_text_to_textlist function
def test_blocking_text_to_textlist(sample_text_to_textlist_dict):
    token_dict = copy.deepcopy(sample_text_to_textlist_dict)
    result = blocking_text_to_textlist(
        token_dict=token_dict,
        block_size=5,
        model_max_length=10,
        pad_token_id=0,
        padding_side="right"
    )
    
    # Check if all sequences are padded to length 5
    assert len(result["input_ids"][0][0]) == 5
    assert len(result["input_ids"][0][1]) == 5
    assert len(result["input_ids"][1][0]) == 5
    assert len(result["input_ids"][1][1]) == 5
    
    # Check first example, first output
    assert result["input_ids"][0][0] == [1, 2, 3, 0, 0]
    
    # Check first example, second output
    assert result["input_ids"][0][1] == [4, 5, 6, 0, 0]

def test_blocking_text_to_textlist_truncation(sample_text_to_textlist_dict):
    # Modify the token_dict to have longer sequences
    token_dict = copy.deepcopy(sample_text_to_textlist_dict)
    token_dict["input_ids"][0][0] = [1, 2, 3, 4, 5, 6]
    
    result = blocking_text_to_textlist(
        token_dict=token_dict,
        block_size=4,
        model_max_length=10,
        pad_token_id=0,
        padding_side="right"
    )
    
    # Check truncation
    assert result["input_ids"][0][0] == [1, 2, 3, 4]
    
    
    
    
##############################################################################################################
################ Tests for tokenize_function in hf_decoder_model and hf_text_regression_model ################
################ Tests for tokenize_function in hf_decoder_model and hf_text_regression_model ################
################ Tests for tokenize_function in hf_decoder_model and hf_text_regression_model ################
################ Tests for tokenize_function in hf_decoder_model and hf_text_regression_model ################
################ Tests for tokenize_function in hf_decoder_model and hf_text_regression_model ################
##############################################################################################################
@patch("transformers.testing_utils.CaptureLogger")
def test_tokenize_function(mock_capture_logger, mock_tokenizer, sample_examples, data_args):
    # Setup mock tokenizer
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]]
    }
    
    # Test with text_only (no label columns)
    result = tokenize_function(
        examples=sample_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["text"],
        label_columns=[],
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check structure of result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Labels should be -100 since text is not in label_columns
    assert result["labels"][0] == [-100, -100, -100]
    assert result["labels"][1] == [-100, -100, -100]

@patch("transformers.testing_utils.CaptureLogger")
def test_tokenize_function_with_labels(mock_capture_logger, mock_tokenizer, sample_examples, data_args):
    # Setup mock tokenizer
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]]
    }
    
    # Test with text2text (with label columns)
    result = tokenize_function(
        examples=sample_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["input", "output"],
        label_columns=["output"],
        tokenized_column_order=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check that labels for output are preserved
    assert result["labels"][0] == [-100, -100, -100, 1, 2, 3]
    assert result["labels"][1] == [-100, -100, -100, 4, 5, 6]

@patch("transformers.testing_utils.CaptureLogger")
def test_tokenize_function_with_blocking(mock_capture_logger, mock_tokenizer, sample_examples, data_args):
    # Setup mock tokenizer
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    }
    
    # Enable blocking
    data_args.disable_group_texts = True
    data_args.block_size = 4
    
    # Test with truncation
    result = tokenize_function(
        examples=sample_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["text"],
        label_columns=[],
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check that blocking was applied (truncation)
    assert len(result["input_ids"][0]) == 4
    assert len(result["attention_mask"][0]) == 4
    assert len(result["labels"][0]) == 4

# Tests for conversation_tokenize_function in hf_decoder_model
@patch("transformers.testing_utils.CaptureLogger")
def test_conversation_tokenize_function_template_obj(
    mock_capture_logger, mock_tokenizer, mock_conversation_template, 
    sample_conversation_examples, data_args
):
    # Test with conversation template object
    result = conversation_tokenize_function(
        examples=sample_conversation_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["messages"],
        conversation_template=mock_conversation_template
    )
    
    # Check result structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Check that mock template was called
    mock_conversation_template.encode_conversation.assert_called()
    
    # Check labels have -100 for user tokens
    assert result["labels"][0] == [-100, -100, -100, 4, 5]
    assert result["labels"][1] == [-100, -100, -100, 4, 5]

@patch("transformers.testing_utils.CaptureLogger")
def test_conversation_tokenize_function_template_str(
    mock_capture_logger, mock_tokenizer,
    sample_conversation_examples, data_args
):
    # Mock the apply_chat_template method
    mock_tokenizer.apply_chat_template.return_value = {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "assistant_masks": [0, 0, 0, 1, 1]  # Last two tokens are from assistant
    }
    
    # Test with string template
    result = conversation_tokenize_function(
        examples=sample_conversation_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["messages"],
        conversation_template="template_string"
    )
    
    # Check tokenizer was called with template
    mock_tokenizer.apply_chat_template.assert_called()
    
    # Check labels have -100 for user tokens
    assert result["labels"][0][:3] == [-100, -100, -100]  # First 3 tokens (user)
    assert result["labels"][0][3:] == [4, 5]  # Last 2 tokens (assistant)

@patch("transformers.testing_utils.CaptureLogger")
def test_conversation_tokenize_function_train_on_prompt(
    mock_capture_logger, mock_tokenizer, mock_conversation_template, 
    sample_conversation_examples, data_args_with_train_on_prompt
):
    # Test with train_on_prompt=True
    result = conversation_tokenize_function(
        examples=sample_conversation_examples,
        data_args=data_args_with_train_on_prompt,
        tokenizer=mock_tokenizer,
        column_names=["messages"],
        conversation_template=mock_conversation_template
    )
    
    # For train_on_prompt=True, labels should include user tokens too
    assert result["labels"][0] == [1, 2, 3, 4, 5]  # All tokens included in labels
    assert result["labels"][1] == [1, 2, 3, 4, 5]

@patch("transformers.testing_utils.CaptureLogger")
def test_conversation_tokenize_function_with_blocking(
    mock_capture_logger, mock_tokenizer, mock_conversation_template, 
    sample_conversation_examples, data_args
):
    # Enable blocking
    data_args.disable_group_texts = True
    data_args.block_size = 3
    
    # Test with blocking enabled
    result = conversation_tokenize_function(
        examples=sample_conversation_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["messages"],
        conversation_template=mock_conversation_template
    )
    
    # Check blocking was applied
    assert len(result["input_ids"][0]) == 3
    assert len(result["attention_mask"][0]) == 3
    assert len(result["labels"][0]) == 3

# Tests for paired_conversation_tokenize_function
@patch("transformers.testing_utils.CaptureLogger")
def test_paired_conversation_tokenize_function(
    mock_capture_logger, mock_tokenizer, mock_conversation_template, 
    sample_paired_conversation_examples, data_args
):
    # Test paired conversation tokenization
    result = paired_conversation_tokenize_function(
        examples=sample_paired_conversation_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["col1", "col2"],
        conversation_template=mock_conversation_template
    )
    
    # Check results structure
    assert "input_ids_col1" in result
    assert "attention_mask_col1" in result
    assert "input_ids_col2" in result
    assert "attention_mask_col2" in result
    
    # Check mock template was called for both columns
    assert mock_conversation_template.encode_conversation.call_count >= 2

@patch("transformers.testing_utils.CaptureLogger")
def test_paired_conversation_tokenize_function_with_error(
    mock_capture_logger, mock_tokenizer, mock_conversation_template, 
    sample_paired_conversation_examples, data_args
):
    # Make encode_conversation raise an error the first time
    def side_effect(*args, **kwargs):
        if side_effect.call_count == 1:
            side_effect.call_count += 1
            raise Exception("Mock error")
        return [([1, 2, 3], [4, 5])]
    
    side_effect.call_count = 0
    mock_conversation_template.encode_conversation.side_effect = side_effect
    
    # Test paired conversation tokenization with error
    result = paired_conversation_tokenize_function(
        examples=sample_paired_conversation_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["col1", "col2"],
        conversation_template=mock_conversation_template
    )
    
    # Function should continue despite error
    assert "input_ids_col1" in result
    assert "input_ids_col2" in result

# Tests for text_to_textlist_tokenize_function
def test_text_to_textlist_tokenize_function(mock_tokenizer, sample_examples, data_args):
    # Setup mock tokenizer
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]]
    }
    
    # Modify sample_examples to have expected structure
    sample_examples["output"] = [["Output 1-1", "Output 1-2"], ["Output 2-1", "Output 2-2"]]
    
    # Test function
    result = text_to_textlist_tokenize_function(
        examples=sample_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check result structure
    assert "input" in result
    assert "output" in result
    assert "input_ids" in result
    
    # Check that tokenizer was called for each output
    assert len(result["input_ids"]) == 2  # Two examples
    assert len(result["input_ids"][0]) == 2  # Two outputs for first example

def test_text_to_textlist_tokenize_function_with_blocking(mock_tokenizer, sample_examples, data_args):
    # Setup mock tokenizer
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    }
    
    # Enable blocking
    data_args.disable_group_texts = True
    data_args.block_size = 3
    
    # Modify sample_examples to have expected structure
    sample_examples["output"] = [["Output 1-1", "Output 1-2"], ["Output 2-1", "Output 2-2"]]
    
    # Test function
    result = text_to_textlist_tokenize_function(
        examples=sample_examples,
        data_args=data_args,
        tokenizer=mock_tokenizer,
        column_names=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check blocking was applied
    assert len(result["input_ids"][0][0]) == 3  # Truncated to block_size
    assert len(result["input_ids"][0][1]) == 3