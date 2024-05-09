#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplate


DEEPSEEK_TEMPLATE = ConversationTemplate(
    template_name='deepseek',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='User: {{content}}\n\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='Assistant: {{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}\n\n')
        ]
    ),
    special_starter=TemplateComponent(type='token', content='bos_token')
)