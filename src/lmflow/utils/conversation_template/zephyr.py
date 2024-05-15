#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from .base import StringFormatter, TemplateComponent, ConversationTemplate


logger = logging.getLogger(__name__)


class ZephyrConversationTemplate(ConversationTemplate):    
    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:
        # TODO: truncation according to model max length
        # TODO: make sure the last few tokens are "learnable", not masked with token_id = -100.
        
        res_all = []
        
        system_formatted = self.system_formatter.format(content=system) if system else []
        system_encoded = self._encode_template(system_formatted, tokenizer)
        
        for i in range(0, len(messages), 2):
            user_message = messages[i]
            assistant_message = messages[i + 1]
            
            user_formatted = self.user_formatter.format(content=user_message["content"])
            if i == 0 and not system:
                # when system is not provided, the first user message should not start with a newline
                user_formatted[0].content = user_formatted[0].content.replace('\n', '', 1)
            assistant_formatted = self.assistant_formatter.format(content=assistant_message["content"])
            
            user_encoded = self._encode_template(user_formatted, tokenizer)
            assistant_encoded = self._encode_template(assistant_formatted, tokenizer)
            
            res_all.append((
                system_encoded + user_encoded if i == 0 else user_encoded, 
                assistant_encoded
            ))
            
        return res_all
    
    
ZEPHYR_TEMPLATE = ZephyrConversationTemplate(
    template_name='zephyr',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='\n<|user|>\n{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='\n<|assistant|>\n{{content}}'),
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