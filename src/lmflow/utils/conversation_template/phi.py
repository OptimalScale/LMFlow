#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplate


PHI3_TEMPLATE = ConversationTemplate(
    template_name='phi3',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|user|>\n{{content}}<|end|>\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|assistant|>\n{{content}}<|end|>\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|system|>\n{{content}}<|end|>\n')
        ]
    ),
    special_starter=TemplateComponent(type='token', content='bos_token'),
    special_stopper=TemplateComponent(type='token', content='eos_token')
)