#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplate


CHATGLM3_TEMPLATE = ConversationTemplate(
    template_name='chatglm3',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|user|>\n {{content}}')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|assistant|>\n {{content}}')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|system|>\n {{content}}')
        ]
    ),
    special_starter=TemplateComponent(type='string', content='[gMASK]sop')
)