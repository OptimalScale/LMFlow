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