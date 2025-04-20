#!/usr/bin/env python
# coding=utf-8
# Additional comparison tests focusing on advanced features and edge cases

import pytest
import torch
import copy
import os
from typing import Dict, List, Union, Optional

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from transformers.testing_utils import CaptureLogger

from lmflow.args import DatasetArguments
from lmflow.tokenization.hf_decoder_model import (
    tokenize_function, 
    conversation_tokenize_function,
    blocking
)
from lmflow.tokenization.hf_text_regression_model import (
    paired_conversation_tokenize_function,
    text_to_textlist_tokenize_function
)
from lmflow.utils.conversation_template import PRESET_TEMPLATES, ConversationTemplate

# Tests for mixed language content
MULTILINGUAL_TEXTS = [
    "English and español mixed together",
    "English and 中文 in the same text",
    "English, français, Deutsch, 日本語, русский all in one text",
    "A real example: J'aime l'intelligence artificielle et 我喜欢人工智能"
]

# Texts with varying special tokens
SPECIAL_TOKEN_TEXTS = [
    "<s>Text with bos token</s>",
    "<pad>Text with pad token</pad>",
    "<eos>Text with eos token</eos>",
    "[CLS]Text with bert tokens[SEP]",
    "<|endoftext|>GPT-style special token",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
    "<|user|>Hello<|assistant|>Hi there!"
]

# Texts that could cause boundary issues with tokenization
BOUNDARY_CASE_TEXTS = [
    "normal.word separated.by.dots",
    "hyphenated-words-with-many-hyphens",
    "under_scored_variable_names",
    "URLs: https://example.com/path?query=string#fragment",
    "Email addresses: user.name+tag@example.co.uk",
    "Special chars at boundaries: !start and end!",
    "Mixed punctuation marks: .,;:!?\"'()[]{}/\\@#$%^&*_+",
    "Numbers and math: 3.14159 + 2.71828 = 5.85987",
    "Code snippets: `import numpy as np; x = np.array([1, 2, 3])`",
    "XML/HTML tags: <div class=\"example\">content</div>"
]

# Text with escaped characters
ESCAPED_CHAR_TEXTS = [
    "Text with newlines: line1\\nline2\\nline3",
    "Text with tabs: col1\\tcol2\\tcol3",
    "Text with quotes: She said, \\\"Hello!\\\"",
    "Text with backslashes: C:\\\\path\\\\to\\\\file.txt",
    "JSON string: \"{\\\"key\\\": \\\"value\\\"}\"",
    "Regex pattern: \"\\\\b\\\\w+\\\\b\"",
    "Unicode escapes: \\u0048\\u0065\\u006C\\u006C\\u006F"
]

# Tests for data structures with many items
LARGE_DATASET = {
    "text": ["Sample text " + str(i) for i in range(100)]
}

# Complex nested conversations
COMPLEX_CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "Hello, I have a question about Python programming."},
            {"role": "assistant", "content": "Of course! I'd be happy to help with your Python question. What would you like to know?"},
            {"role": "user", "content": "How do I handle JSON data in Python?"},
            {"role": "assistant", "content": "Working with JSON in Python is straightforward using the built-in `json` module.\n\n```python\nimport json\n\n# Parse JSON from string\ndata = json.loads('{\"name\": \"John\", \"age\": 30}')\nprint(data['name'])  # Output: John\n\n# Convert Python object to JSON string\npython_obj = {\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}\njson_string = json.dumps(python_obj, indent=4)\nprint(json_string)\n```\n\nYou can also read and write JSON files:\n\n```python\n# Writing JSON to a file\nwith open('data.json', 'w') as f:\n    json.dump(python_obj, f, indent=4)\n\n# Reading JSON from a file\nwith open('data.json', 'r') as f:\n    loaded_data = json.load(f)\n```\n\nIs there a specific aspect of working with JSON that you're interested in?"},
            {"role": "user", "content": "How do I handle nested JSON objects?"},
            {"role": "assistant", "content": "Handling nested JSON objects in Python is quite simple because JSON naturally maps to Python's data structures. Nested objects become nested dictionaries, and arrays become lists.\n\nHere's an example of working with nested JSON:\n\n```python\nimport json\n\n# A nested JSON structure\nnested_json = '''\n{\n    \"person\": {\n        \"name\": \"John\",\n        \"age\": 30,\n        \"address\": {\n            \"street\": \"123 Main St\",\n            \"city\": \"Anytown\",\n            \"zipcode\": \"12345\"\n        },\n        \"phones\": [\n            {\"type\": \"home\", \"number\": \"555-1234\"},\n            {\"type\": \"work\", \"number\": \"555-5678\"}\n        ]\n    }\n}\n'''\n\n# Parse the JSON\ndata = json.loads(nested_json)\n\n# Access nested fields using multiple keys\nprint(data['person']['name'])  # Output: John\nprint(data['person']['address']['city'])  # Output: Anytown\n\n# Access array elements\nprint(data['person']['phones'][0]['number'])  # Output: 555-1234\n\n# Modify nested data\ndata['person']['address']['zipcode'] = \"54321\"\n\n# Add new nested data\ndata['person']['email'] = \"john@example.com\"\n\n# Convert back to JSON\nupdated_json = json.dumps(data, indent=4)\nprint(updated_json)\n```\n\nFor complex JSON structures, you might want to consider using dot notation libraries like `jmespath` or `jsonpath-ng` that make it easier to query nested data.\n\nIs there a specific scenario with nested JSON that you're working with?"}
        ],
        "system": "You are a helpful programming assistant."
    }
]

# Text-to-textlist examples for RM training
COMPLEX_TEXT_TO_TEXTLIST = {
    "input": [
        "Rank the following responses from best to worst:",
        "Which of these SQL queries will perform better?",
        "Which translation is most natural?"
    ],
    "output": [
        [
            "Response A: I'll help you solve this complex problem by breaking it down into manageable steps. First, let's identify the key variables involved...",
            "Response B: The answer is 42.",
            "Response C: I don't know how to help with this. Maybe try Google?",
            "Response D: I'm sorry, but I can't assist with this specific task because it involves potentially harmful content."
        ],
        [
            "Query 1: SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE orders.status = 'completed';",
            "Query 2: SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id WHERE orders.status = 'completed';",
            "Query 3: SELECT users.*, orders.* FROM users, orders WHERE users.id = orders.user_id AND orders.status = 'completed';"
        ],
        [
            "Translation 1: Je suis très heureux de vous rencontrer.",
            "Translation 2: Je suis extrêmement heureux de faire votre connaissance.",
            "Translation 3: C'est une grande joie pour moi de vous rencontrer finalement."
        ]
    ]
}

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
def data_args_small_blocks():
    """Create DatasetArguments with small blocks."""
    return DatasetArguments(
        dataset_path=None,
        disable_group_texts=True,
        block_size=32,
        train_on_prompt=False
    )

@pytest.mark.parametrize("model_name", ["gpt2", "google/mt5-small", "xlm-roberta-base"])
def test_multilingual_tokenization(model_name, data_args):
    """Test tokenization with multilingual content."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer for {model_name}: {str(e)}")
        
    # If the tokenizer doesn't have a pad token, set it to eos_token
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples
    examples = {"text": MULTILINGUAL_TEXTS}
    
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
    for i in range(len(examples["text"])):
        if isinstance(hf_encoding["input_ids"][i], list):
            hf_ids = hf_encoding["input_ids"][i]
        else:
            hf_ids = hf_encoding["input_ids"][i].tolist()
            
        lmflow_ids = lmflow_encoding["input_ids"][i]
        
        # Print some debug info if they don't match
        if hf_ids != lmflow_ids:
            print(f"Mismatch with multilingual text at index {i}:")
            print(f"Text: {examples['text'][i]}")
            print(f"HF tokens    : {hf_ids}")
            print(f"LMFlow tokens: {lmflow_ids}")
            print(f"HF decoded   : {tokenizer.decode(hf_ids)}")
            print(f"LMFlow decoded: {tokenizer.decode(lmflow_ids)}")
            
        assert hf_ids == lmflow_ids, f"Multilingual tokenization mismatch at index {i}"

@pytest.mark.parametrize("model_name", ["gpt2", "bert-base-uncased"])
def test_special_token_handling(model_name, data_args):
    """Test how special tokens are handled by tokenizers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If the tokenizer doesn't have a pad token, set it to eos_token
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples
    examples = {"text": SPECIAL_TOKEN_TEXTS}
    
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
    for i in range(len(examples["text"])):
        if isinstance(hf_encoding["input_ids"][i], list):
            hf_ids = hf_encoding["input_ids"][i]
        else:
            hf_ids = hf_encoding["input_ids"][i].tolist()
            
        lmflow_ids = lmflow_encoding["input_ids"][i]
        
        assert hf_ids == lmflow_ids, f"Special token handling mismatch at index {i}"

def test_tokenization_boundary_cases(data_args):
    """Test tokenization with text that might cause boundary issues."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare examples
    examples = {"text": BOUNDARY_CASE_TEXTS}
    
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
    for i in range(len(examples["text"])):
        if isinstance(hf_encoding["input_ids"][i], list):
            hf_ids = hf_encoding["input_ids"][i]
        else:
            hf_ids = hf_encoding["input_ids"][i].tolist()
            
        lmflow_ids = lmflow_encoding["input_ids"][i]
        
        assert hf_ids == lmflow_ids, f"Boundary case tokenization mismatch at index {i}"

def test_escaped_character_handling(data_args):
    """Test tokenization with escaped characters."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare examples - need to handle raw strings for proper escaping
    raw_examples = [
        r"Text with newlines: line1\nline2\nline3",
        r"Text with tabs: col1\tcol2\tcol3",
        r"Text with quotes: She said, \"Hello!\"",
        r"Text with backslashes: C:\\path\\to\\file.txt",
        r"JSON string: \"{\\\"key\\\": \\\"value\\\"}\"",
        r"Regex pattern: \"\\b\\w+\\b\"",
        r"Unicode escapes: \u0048\u0065\u006C\u006C\u006F"  # Should be "Hello"
    ]
    
    examples = {"text": raw_examples}
    
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
    for i in range(len(examples["text"])):
        if isinstance(hf_encoding["input_ids"][i], list):
            hf_ids = hf_encoding["input_ids"][i]
        else:
            hf_ids = hf_encoding["input_ids"][i].tolist()
            
        lmflow_ids = lmflow_encoding["input_ids"][i]
        
        if hf_ids != lmflow_ids:
            print(f"Mismatch with escaped chars at index {i}:")
            print(f"Text: {examples['text'][i]}")
            print(f"HF decoded   : {tokenizer.decode(hf_ids)}")
            print(f"LMFlow decoded: {tokenizer.decode(lmflow_ids)}")
            
        assert hf_ids == lmflow_ids, f"Escaped char tokenization mismatch at index {i}"

def test_large_dataset_handling():
    """Test tokenization with a larger dataset."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token for GPT-2 tokenizer (which doesn't have one by default)
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data args for small blocks to make processing faster
    data_args = DatasetArguments(
        dataset_path=None,
        disable_group_texts=True,
        block_size=10  # Small blocks for faster processing
    )
    
    # Original HuggingFace tokenization
    hf_encoding = tokenizer(
        LARGE_DATASET["text"],
        add_special_tokens=True,
        truncation=True,
        max_length=10,  # Match block_size
        padding="max_length",  # Pad to max_length
        return_tensors=None
    )
    
    # LMFlow tokenization
    lmflow_encoding = tokenize_function(
        examples=LARGE_DATASET,
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=["text"],
        label_columns=["text"],
        tokenized_column_order=["text"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check structure
    assert len(lmflow_encoding["input_ids"]) == len(LARGE_DATASET["text"]), "Number of examples mismatch"
    
    # Check a few random examples
    import random
    random.seed(42)  # For reproducibility
    
    for _ in range(10):  # Check 10 random examples
        i = random.randint(0, len(LARGE_DATASET["text"]) - 1)
        
        if isinstance(hf_encoding["input_ids"][i], list):
            hf_ids = hf_encoding["input_ids"][i]
        else:
            hf_ids = hf_encoding["input_ids"][i].tolist()
            
        lmflow_ids = lmflow_encoding["input_ids"][i]
        
        assert len(lmflow_ids) == data_args.block_size, f"Block size mismatch at index {i}"
        assert hf_ids == lmflow_ids, f"Tokenization mismatch at index {i}"

def test_complex_conversation_tokenization(data_args):
    """Test tokenization with complex multi-turn conversations."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get a simple template
    template = PRESET_TEMPLATES["empty_no_special_tokens"]
    
    # For each complex conversation
    for i, conversation in enumerate(COMPLEX_CONVERSATIONS):
        # Prepare examples for LMFlow
        examples = {
            "messages": [conversation["messages"]],
            "system": [conversation["system"]],
            "tools": [None]
        }
        
        # Use LMFlow conversation tokenization
        lmflow_encoding = conversation_tokenize_function(
            examples=examples,
            data_args=data_args,
            tokenizer=tokenizer,
            column_names=["messages"],
            conversation_template=template
        )
        
        # Check structure
        assert "input_ids" in lmflow_encoding, "input_ids missing from encoding"
        assert "attention_mask" in lmflow_encoding, "attention_mask missing from encoding"
        assert "labels" in lmflow_encoding, "labels missing from encoding"
        
        # Check non-empty sequences
        assert len(lmflow_encoding["input_ids"][0]) > 0, "input_ids should not be empty"
        
        # Check that we have the expected number of -100 values in labels
        # For each user message, we should have a section of -100s
        user_token_count = sum(len(tokenizer.encode(msg["content"])) 
                              for msg in conversation["messages"] 
                              if msg["role"] == "user")
        
        neg_100_count = lmflow_encoding["labels"][0].count(-100)
        assert neg_100_count > 0, "Should have some -100 values for user messages"
        
        # Decode the input_ids to check the full text
        full_text = tokenizer.decode(lmflow_encoding["input_ids"][0])
        
        # Check that the text contains content from each message
        for msg in conversation["messages"]:
            # Just check for a substring of each message to avoid exact format matching
            content_snippet = msg["content"][:20]  # First 20 chars
            assert content_snippet in full_text, f"Message content not found in tokenized output: {content_snippet}"

def test_text_to_textlist_complex(data_args_small_blocks):
    """Test tokenization with complex text_to_textlist examples."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token for GPT-2 tokenizer
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use LMFlow text_to_textlist tokenization
    result = text_to_textlist_tokenize_function(
        examples=COMPLEX_TEXT_TO_TEXTLIST,
        data_args=data_args_small_blocks,
        tokenizer=tokenizer,
        column_names=["input", "output"],
        add_special_tokens=True,
        use_truncation=True
    )
    
    # Check structure
    assert "input" in result, "input missing from result"
    assert "output" in result, "output missing from result"
    assert "input_ids" in result, "input_ids missing from result"
    
    # Check dimensions
    assert len(result["input_ids"]) == len(COMPLEX_TEXT_TO_TEXTLIST["input"]), "Number of inputs mismatch"
    
    # Check structure for each example
    for i in range(len(result["input_ids"])):
        assert len(result["input_ids"][i]) == len(COMPLEX_TEXT_TO_TEXTLIST["output"][i]), "Number of outputs mismatch"
        
        # Check block size for each choice
        for j in range(len(result["input_ids"][i])):
            # First check if there are None values in the input_ids
            assert result["input_ids"][i][j] is not None, f"input_ids[{i}][{j}] is None"
            
            # Check for None values within the token list
            assert None not in result["input_ids"][i][j], f"None found in input_ids[{i}][{j}]"
            
            assert len(result["input_ids"][i][j]) == data_args_small_blocks.block_size, "Block size mismatch"
            
            # Decode to check content (with error handling)
            try:
                decoded = tokenizer.decode(result["input_ids"][i][j])
                
                # Continue with checking content
                input_snippet = COMPLEX_TEXT_TO_TEXTLIST["input"][i][:15]  # First 15 chars
                output_snippet = COMPLEX_TEXT_TO_TEXTLIST["output"][i][j][:15]  # First 15 chars
                
                # Check that both input and output snippets are in the decoded text
                assert input_snippet in decoded, f"Input snippet not found in tokenized output: {input_snippet}"
                assert output_snippet in decoded, f"Output snippet not found in tokenized output: {output_snippet}"
            except Exception as e:
                pytest.fail(f"Error decoding tokens at input_ids[{i}][{j}]: {str(e)}\nTokens: {result['input_ids'][i][j]}")

def test_paired_conversation_complex():
    """Test paired_conversation_tokenize_function with complex examples."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token for GPT-2 tokenizer
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data args
    data_args = DatasetArguments(
        dataset_path=None,
        disable_group_texts=True,
        block_size=64  # Small blocks for testing
    )
    
    # Get a simple template
    template = PRESET_TEMPLATES["empty_no_special_tokens"]
    
    # Create paired examples from our complex conversations
    paired_examples = {
        "chosen": [{"messages": conv["messages"], "system": conv["system"]} for conv in COMPLEX_CONVERSATIONS],
        "rejected": [{"messages": conv["messages"][:2], "system": conv["system"]} for conv in COMPLEX_CONVERSATIONS]
    }
    
    # Use paired_conversation_tokenize_function
    result = paired_conversation_tokenize_function(
        examples=paired_examples,
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=["chosen", "rejected"],
        conversation_template=template
    )
    
    # Check structure
    assert "input_ids_chosen" in result, "input_ids_chosen missing"
    assert "attention_mask_chosen" in result, "attention_mask_chosen missing"
    assert "input_ids_rejected" in result, "input_ids_rejected missing"
    assert "attention_mask_rejected" in result, "attention_mask_rejected missing"
    
    # Check block size
    for i in range(len(result["input_ids_chosen"])):
        # Check for None values
        assert result["input_ids_chosen"][i] is not None, f"input_ids_chosen[{i}] is None"
        assert None not in result["input_ids_chosen"][i], f"None found in input_ids_chosen[{i}]"
        
        assert result["input_ids_rejected"][i] is not None, f"input_ids_rejected[{i}] is None"
        assert None not in result["input_ids_rejected"][i], f"None found in input_ids_rejected[{i}]"
        
        assert len(result["input_ids_chosen"][i]) == data_args.block_size, "Block size mismatch for chosen"
        assert len(result["input_ids_rejected"][i]) == data_args.block_size, "Block size mismatch for rejected"
        
        # Decode to check content with error handling
        try:
            # Skip special tokens to get clean text for comparison
            chosen_text = tokenizer.decode(result["input_ids_chosen"][i], skip_special_tokens=True)
            rejected_text = tokenizer.decode(result["input_ids_rejected"][i], skip_special_tokens=True)
            
            # Check non-empty texts
            assert len(chosen_text) > 0, "Chosen text should not be empty"
            assert len(rejected_text) > 0, "Rejected text should not be empty"
            
            # Check specific content from the first user message
            first_user_msg = COMPLEX_CONVERSATIONS[i]["messages"][0]["content"][:20]
            if first_user_msg not in chosen_text:
                print(f"Warning: First user message not found in chosen text.")
                print(f"Message: {first_user_msg}")
                print(f"Chosen text: {chosen_text[:100]}...")
            if first_user_msg not in rejected_text:
                print(f"Warning: First user message not found in rejected text.")
                print(f"Message: {first_user_msg}")
                print(f"Rejected text: {rejected_text[:100]}...")
                
            # Check specific content instead of just comparing lengths
            if len(COMPLEX_CONVERSATIONS[i]["messages"]) > 2:
                # Get content from third message (should be in chosen but not rejected)
                third_message_snippet = COMPLEX_CONVERSATIONS[i]["messages"][2]["content"][:20] 
                
                # Check if this content is in chosen text (it should be)
                if third_message_snippet not in chosen_text:
                    print(f"Warning: Third message content not found in chosen text where it should be.")
                    print(f"Third message snippet: {third_message_snippet}")
                    print(f"Chosen text: {chosen_text}")
                
                # We don't assert here since truncation might cut off later messages
        except Exception as e:
            pytest.fail(f"Error decoding tokens at index {i}: {str(e)}\n"
                      f"Chosen tokens: {result['input_ids_chosen'][i][:10]}...\n"
                      f"Rejected tokens: {result['input_ids_rejected'][i][:10]}...")

def test_different_padding_sides():
    """Test tokenization with different padding sides."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test both padding sides
    for padding_side in ["left", "right"]:
        # Set tokenizer padding side
        tokenizer.padding_side = padding_side
        
        # Set pad_token for GPT-2 tokenizer
        if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create data args
        data_args = DatasetArguments(
            dataset_path=None,
            disable_group_texts=True,
            block_size=20
        )
        
        # Create examples with varying lengths
        texts = ["Short", "Medium length text", "This is a longer text that will need padding"]
        examples = {"text": texts}
        
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
        
        # Check each example
        for i, text in enumerate(texts):
            # Direct tokenization for comparison
            direct_ids = tokenizer.encode(
                text, 
                add_special_tokens=True,
                truncation=True,
                max_length=data_args.block_size,
                padding="max_length"
            )
            
            lmflow_ids = lmflow_encoding["input_ids"][i]
            
            if direct_ids != lmflow_ids:
                print(f"Mismatch with padding_side={padding_side}, text: '{text}'")
                print(f"Direct: {direct_ids}")
                print(f"LMFlow: {lmflow_ids}")
                print(f"Direct decoded: {tokenizer.decode(direct_ids)}")
                print(f"LMFlow decoded: {tokenizer.decode(lmflow_ids)}")
            
            assert direct_ids == lmflow_ids, f"Mismatch with padding_side={padding_side}"
            
            # Check block size
            assert len(lmflow_ids) == data_args.block_size, f"Block size mismatch with padding_side={padding_side}"
            
            # Check padding location based on padding_side
            text_tokens = tokenizer.encode(text, add_special_tokens=True)
            pad_token = tokenizer.pad_token_id
            
            if len(text_tokens) < data_args.block_size:
                if padding_side == "right":
                    # Padding should be at the end
                    assert lmflow_ids[-1] == pad_token, "Right padding should end with pad_token"
                    assert lmflow_ids[0] != pad_token, "Right padding should not start with pad_token"
                else:
                    # Padding should be at the beginning
                    assert lmflow_ids[0] == pad_token, "Left padding should start with pad_token"
                    assert lmflow_ids[-1] != pad_token, "Left padding should not end with pad_token"