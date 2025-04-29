#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
from dataclasses import dataclass

from .base import StringFormatter, TemplateComponent, ConversationTemplate


logger = logging.getLogger(__name__)


@dataclass
class GemmaConversationTemplate(ConversationTemplate):
    def encode_conversation(self, *args, **kwargs):
        if kwargs.get('system'):
            logger.warning(
                'As of now, Gemma does not support system messages officially. '
                'ConversationTemplate will add your system messages right after '
                'the bos token and before the user message without any special formatting. '
                'For more details, please refer to the [official template]'
                '(https://huggingface.co/google/gemma-1.1-2b-it/blob/bf4924f313df5166dee1467161e886e55f2eb4d4/tokenizer_config.json#L1507).'
            )
        return super().encode_conversation(*args, **kwargs)
        

GEMMA_TEMPLATE = GemmaConversationTemplate(
    template_name='gemma',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<start_of_turn>user\n{{content}}<end_of_turn>\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<start_of_turn>model\n{{content}}<end_of_turn>\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}')
        ]
    ),
    special_starter=TemplateComponent(type='token', content='bos_token')
)


GEMMA3_TEMPLATE = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}\n{%- else -%}
    {%- set first_user_prefix = \"\" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = \"model\" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}
    {%- if (message['role'] == 'assistant') -%}
        {% generation %}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception(\"Invalid content type\") }}
        {%- endif -%}
        {{ '<end_of_turn>\n' }}
        {% endgeneration %}
    {%- else -%}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception(\"Invalid content type\") }}
        {%- endif -%}
        {{ '<end_of_turn>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""