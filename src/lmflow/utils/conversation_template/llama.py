#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from .base import StringFormatter, TemplateComponent, ConversationTemplate, ConversationTemplateForTool

from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

logger = logging.getLogger(__name__)

class Llama2ConversationTemplate(ConversationTemplate):    
    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:
        if tools:
            logger.warning("Formatted tools are not supported in Llama2, thus tools will be ignored. "
                           "If this is intended, please include tools in the system message manually.")
        
        res_all = []
        
        system_formatted = self.system_formatter.format(content=system) if system else []
        system_formatted_text = "".join([component.content for component in system_formatted if component.type == 'string']) # HACK
        
        for i in range(0, len(messages), 2):
            user_message = messages[i]
            assistant_message = messages[i + 1]
            
            user_content = system_formatted_text + user_message["content"] if i == 0 else user_message["content"]
            user_formatted = self.user_formatter.format(content=user_content)
            assistant_formatted = self.assistant_formatter.format(content=assistant_message["content"])
            
            user_encoded = self._encode_template(user_formatted, tokenizer)
            assistant_encoded = self._encode_template(assistant_formatted, tokenizer)
            
            res_all.append((
                user_encoded, 
                assistant_encoded
            ))
            
        return res_all

class Llama2ConversationTemplateForTool(Llama2ConversationTemplate):    
    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:
        if tools:
            # logger.warning("Formatted tools are not supported in Llama2, thus tools will be ignored. "
            #                "If this is intended, please include tools in the system message manually.")
            system = system + tools
        res_all = []
        system_formatted = self.system_formatter.format(content=system) if system else []
        system_formatted_text = "".join([component.content for component in system_formatted if component.type == 'string']) # HACK
        ls_for_save = []
        for i in range(0, len(messages), 1):
            if messages[i]['role'] == CONVERSATION_ROLE_NAMES['user']:
                user_message = messages[i]
                if i == 0:
                    user_content = system_formatted_text + user_message['content']
                else:
                    user_content = user_message['content']
                user_formatted = self.user_formatter.format(content=user_content)
                user_encoded = self._encode_template(user_formatted, tokenizer)
                ls_for_save.append(user_encoded)
            elif messages[i]['role'] == CONVERSATION_ROLE_NAMES['function']:
                function_message = messages[i]
                function_formatted = self.assistant_formatter.format(content=function_message['content'])
                function_encoded = self._encode_template(function_formatted, tokenizer)
                ls_for_save.append(function_encoded)
            elif messages[i]['role'] == CONVERSATION_ROLE_NAMES['observation']:
                observation_message = messages[i]
                observation_formatted = self.user_formatter.format(content=observation_message['content'])
                observation_encoded = self._encode_template(observation_formatted, tokenizer)
                ls_for_save.append(observation_encoded)
            elif messages[i]['role'] == CONVERSATION_ROLE_NAMES['assistant']:
                assistant_message = messages[i]
                assistant_formatted = self.assistant_formatter.format(content=assistant_message["content"])
                assistant_encoded = self._encode_template(assistant_formatted, tokenizer)
                ls_for_save.append(assistant_encoded)
                # res_tuple = (ls_for_save[0], ls_for_save[1], ls_for_save[2], ls_for_save[3])
                res_all.append(tuple(ls_for_save))
                ls_for_save = []
        
        if ls_for_save:
            res_all.append(tuple(ls_for_save))

        return res_all

LLAMA3_TEMPLATE = ConversationTemplate(
    template_name='llama3',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>assistant<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    special_starter=TemplateComponent(type='token', content='bos_token')
)

LLAMA3_TEMPLATE_FOR_TOOL = ConversationTemplateForTool(
    template_name='llama3_for_tool',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    function_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>assistant<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    observation_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>assistant<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    ),
    special_starter=TemplateComponent(type='token', content='bos_token')
)


LLAMA2_TEMPLATE = Llama2ConversationTemplate(
    template_name='llama2',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='[INST] {{content}} [/INST]')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<<SYS>>\n{{content}}\n<</SYS>>\n\n')
        ]
    )
)

LLAMA2_TEMPLATE_FOR_TOOL = Llama2ConversationTemplate(
    template_name='llama2_for_tool',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='[INST] {{content}} [/INST]')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<<SYS>>\n{{content}}\n<</SYS>>\n\n')
        ]
    )
)