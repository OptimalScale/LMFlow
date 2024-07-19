#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplate

# {% for message in messages %}
# {% if message['role'] == 'user' %}
# {{ '<|user|>\n' + message['content'] + eos_token }}
# {% elif message['role'] == 'system' %}
# {{ '<|system|>\n' + message['content'] + eos_token }}
# {% elif message['role'] == 'assistant' %}
# {{ '<|assistant|>\n'  + message['content'] + eos_token }}
# {% endif %}
# {% if loop.last and add_generation_prompt %}
# {{ '<|assistant|>' }}
# {% endif %}
# {% endfor %}
FOX_TEMPLATE = ConversationTemplate(
    template_name='fox',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|user|>\n{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|assistant|>\n{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|system|>\n{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    )
)