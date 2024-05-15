#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplate


ZEPHYR_TEMPLATE = ConversationTemplate(
    template_name='zephyr',
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