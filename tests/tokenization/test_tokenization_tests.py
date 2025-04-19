#!/usr/bin/env python
# coding=utf-8
# Tests for conversation template utilities

'''
Tests for the conversation template system, including string formatting and various template presets.
'''

import pytest
from unittest.mock import MagicMock, patch

from transformers import AutoTokenizer

from lmflow.utils.conversation_template import (
    ConversationTemplate,
    PRESET_TEMPLATES,
    JINJA_TEMPLATES,
    StringFormatter,
    TemplateComponent
)
from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

# Sample conversations for testing
SAMPLE_SYSTEM_MESSAGE = "You are a helpful assistant."

SAMPLE_SINGLE_TURN = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
]

SAMPLE_MULTI_TURN = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
    {"role": "user", "content": "Can you help me with a task?"},
    {"role": "assistant", "content": "Of course! I'd be happy to help. What do you need assistance with?"}
]

# Fixtures for testing
@pytest.fixture
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained('gpt2')

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [100 + i for i in range(len(text.split()))]
    return tokenizer

# Tests for StringFormatter
def test_string_formatter_basic():
    # Create a simple formatter with string components
    formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='User: {{content}}'),
            TemplateComponent(type='string', content='\nAssistant:')
        ]
    )
    
    # Format a message
    result = formatter.format(content="Hello world")
    
    # Check the result
    assert len(result) == 2
    assert result[0].type == 'string'
    assert result[0].content == 'User: Hello world'
    assert result[1].type == 'string'
    assert result[1].content == '\nAssistant:'

def test_string_formatter_with_tokens():
    # Create a formatter with token components
    formatter = StringFormatter(
        template=[
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='User: {{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    )
    
    # Format a message
    result = formatter.format(content="Hello world")
    
    # Check the result
    assert len(result) == 3
    assert result[0].type == 'token'
    assert result[0].content == 'bos_token'
    assert result[1].type == 'string'
    assert result[1].content == 'User: Hello world'
    assert result[2].type == 'token'
    assert result[2].content == 'eos_token'

def test_string_formatter_multiple_variables():
    # Create a formatter with multiple variables
    formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{role}}: {{content}}')
        ]
    )
    
    # Format a message with multiple variables
    result = formatter.format(role="User", content="Hello world")
    
    # Check the result
    assert len(result) == 1
    assert result[0].type == 'string'
    assert result[0].content == 'User: Hello world'

# Tests for ConversationTemplate
def test_conversation_template_encode_single_message(mock_tokenizer):
    # Create a simple template
    template = ConversationTemplate(
        user_formatter=StringFormatter([TemplateComponent(type='string', content='User: {{content}}')]),
        assistant_formatter=StringFormatter([TemplateComponent(type='string', content='Assistant: {{content}}')])
    )
    
    # Encode a single turn conversation
    result = template.encode_conversation(
        tokenizer=mock_tokenizer,
        messages=SAMPLE_SINGLE_TURN,
        system=None,
        tools=None
    )
    
    # Check result structure
    assert len(result) == 1  # One turn
    assert len(result[0]) == 2  # (user, assistant) pair
    
    # Check tokens were encoded
    assert isinstance(result[0][0], list)  # User tokens
    assert isinstance(result[0][1], list)  # Assistant tokens
    
    # Mock tokenizer just returns tokens based on number of words
    assert len(result[0][0]) == 3  # "User: Hello how are you?" -> 3 tokens (in our mock)
    assert len(result[0][1]) == 2  # "Assistant: I'm doing..." -> 2 tokens (simplified in mock)

def test_conversation_template_encode_multi_turn(mock_tokenizer):
    # Create a simple template
    template = ConversationTemplate(
        user_formatter=StringFormatter([TemplateComponent(type='string', content='User: {{content}}')]),
        assistant_formatter=StringFormatter([TemplateComponent(type='string', content='Assistant: {{content}}')])
    )
    
    # Encode a multi-turn conversation
    result = template.encode_conversation(
        tokenizer=mock_tokenizer,
        messages=SAMPLE_MULTI_TURN,
        system=None,
        tools=None
    )
    
    # Check result structure
    assert len(result) == 2  # Two turns
    assert len(result[0]) == 2  # First (user, assistant) pair
    assert len(result[1]) == 2  # Second (user, assistant) pair
    
    # Check specific token counts (based on our mock tokenizer)
    assert len(result[0][0]) > 0  # First user turn
    assert len(result[0][1]) > 0  # First assistant turn
    assert len(result[1][0]) > 0  # Second user turn
    assert len(result[1][1]) > 0  # Second assistant turn

def test_conversation_template_with_system_message(mock_tokenizer):
    # Create a template with system formatter
    template = ConversationTemplate(
        system_formatter=StringFormatter([TemplateComponent(type='string', content='System: {{content}}\n')]),
        user_formatter=StringFormatter([TemplateComponent(type='string', content='User: {{content}}')]),
        assistant_formatter=StringFormatter([TemplateComponent(type='string', content='Assistant: {{content}}')])
    )
    
    # Encode with system message
    result = template.encode_conversation(
        tokenizer=mock_tokenizer,
        messages=SAMPLE_SINGLE_TURN,
        system=SAMPLE_SYSTEM_MESSAGE,
        tools=None
    )
    
    # Check result with system included
    assert len(result) == 1  # One turn
    assert len(result[0]) == 2  # (user, assistant) pair
    
    # System should be included in the first user input
    assert len(result[0][0]) > 0  # User tokens with system
    
    # Mock tokenizer doesn't actually handle system token separation, 
    # but in a real tokenizer, system would be part of the first encoded sequence

def test_preset_templates(gpt2_tokenizer):
    # Check that we can access all preset templates
    assert 'empty' in PRESET_TEMPLATES
    assert 'empty_no_special_tokens' in PRESET_TEMPLATES
    assert 'llama2' in PRESET_TEMPLATES
    
    # Test encoding with a preset template
    empty_template = PRESET_TEMPLATES['empty_no_special_tokens']
    
    # Encode a conversation
    result = empty_template.encode_conversation(
        tokenizer=gpt2_tokenizer,
        messages=SAMPLE_SINGLE_TURN,
        system=None,
        tools=None
    )
    
    # Should get a valid encoding
    assert len(result) == 1  # One turn
    assert len(result[0]) == 2  # (user, assistant) pair
    assert isinstance(result[0][0], list)  # User tokens
    assert isinstance(result[0][1], list)  # Assistant tokens
    assert len(result[0][0]) > 0  # User tokens not empty
    assert len(result[0][1]) > 0  # Assistant tokens not empty

def test_conversation_template_invalid_conversation():
    # Create a simple template
    template = ConversationTemplate(
        user_formatter=StringFormatter([TemplateComponent(type='string', content='User: {{content}}')]),
        assistant_formatter=StringFormatter([TemplateComponent(type='string', content='Assistant: {{content}}')])
    )
    
    # Test with invalid conversation (odd number of messages)
    invalid_conversation = SAMPLE_MULTI_TURN[:-1]  # Remove last assistant message
    
    # This should not raise an error, but log a warning
    with patch('lmflow.tokenization.hf_decoder_model.logger.warning') as mock_warning:
        result = template.encode_conversation(
            tokenizer=mock_tokenizer(),
            messages=invalid_conversation,
            system=None,
            tools=None
        )
        
        # Should still get a result with complete turns only
        assert len(result) == 1  # One complete turn only

    # Test with first message not from user
    invalid_first_role = [
        {"role": "assistant", "content": "I'm an assistant"},
        {"role": "user", "content": "Hello"}
    ]
    
    # This should log a warning and skip this conversation
    with patch('lmflow.tokenization.hf_decoder_model.tok_logger.warning') as mock_warning:
        result = template.encode_conversation(
            tokenizer=mock_tokenizer(),
            messages=invalid_first_role,
            system=None,
            tools=None
        )
        
        # Should be skipped or empty
        assert len(result) == 0 or result == []

# Tests for specific templates
def test_llama2_template(gpt2_tokenizer):
    # Get the llama2 template
    llama2_template = PRESET_TEMPLATES['llama2']
    
    # Encode a conversation
    result = llama2_template.encode_conversation(
        tokenizer=gpt2_tokenizer,
        messages=SAMPLE_SINGLE_TURN,
        system=SAMPLE_SYSTEM_MESSAGE,
        tools=None
    )
    
    # Should get a valid encoding
    assert len(result) == 1  # One turn
    assert len(result[0]) == 2  # (user, assistant) pair
    
    # Check that the encoded user input contains the system message
    user_tokens = result[0][0]
    assistant_tokens = result[0][1]
    
    # Decode tokens to check format
    user_text = gpt2_tokenizer.decode(user_tokens)
    assistant_text = gpt2_tokenizer.decode(assistant_tokens)
    
    # User text should contain both system and user message in llama2 format
    assert "SYS" in user_text or "[INST]" in user_text
    
    # Multi-turn test
    multi_result = llama2_template.encode_conversation(
        tokenizer=gpt2_tokenizer,
        messages=SAMPLE_MULTI_TURN,
        system=SAMPLE_SYSTEM_MESSAGE,
        tools=None
    )
    
    # Should get two turns
    assert len(multi_result) == 2
    
    # First turn should include system message, second should not
    first_turn_text = gpt2_tokenizer.decode(multi_result[0][0])
    second_turn_text = gpt2_tokenizer.decode(multi_result[1][0])
    
    assert "SYS" in first_turn_text or "<<SYS>>" in first_turn_text
    # Second turn shouldn't have system prompt
    if "SYS" in second_turn_text:
        assert not ("<<SYS>>\n" + SAMPLE_SYSTEM_MESSAGE in second_turn_text)