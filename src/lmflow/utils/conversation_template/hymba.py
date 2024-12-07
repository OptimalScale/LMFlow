#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplateForTool
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple

from transformers import PreTrainedTokenizer


# {{'<extra_id_0>System'}}
# {% for message in messages %}
#   {% if message['role'] == 'system' %}
#       {{'\n' + message['content'].strip()}}
#       {% if tools or contexts %}
#           {{'\n'}}
#       {% endif %}
#   {% endif %}
# {% endfor %}
# {% if tools %}
#   {% for tool in tools %}
#       {{ '\n<tool> ' + tool|tojson + ' </tool>' }}
#   {% endfor %}
# {% endif %}
# {% if contexts %}
#   {% if tools %}
#       {{'\n'}}
#   {% endif %}
#   {% for context in contexts %}
#       {{ '\n<context> ' + context.strip() + ' </context>' }}
#   {% endfor %}
# {% endif %}
# {{'\n\n'}}
# {% for message in messages %}
#   {% if message['role'] == 'user' %}
#       {{ '<extra_id_1>User\n' + message['content'].strip() + '\n' }}
#   {% elif message['role'] == 'assistant' %}
#       {{ '<extra_id_1>Assistant\n' + message['content'].strip() + '\n' }}
#   {% elif message['role'] == 'tool' %}
#       {{ '<extra_id_1>Tool\n' + message['content'].strip() + '\n' }}
#   {% endif %}
# {% endfor %}
# {%- if add_generation_prompt %}
#   {{'<extra_id_1>Assistant\n'}}
# {%- endif %}


class HymbaConversationTemplate(ConversationTemplateForTool):
    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:        
        system_and_tool = system + "<tool> " + tools + " </tool>"
        return super()._encode(tokenizer=tokenizer,messages=messages, system=system_and_tool, tools='')
    


HYMBA_TEMPLATE = HymbaConversationTemplate(
    template_name='hymba',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<extra_id_1>User\n{{content}}\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<extra_id_1>Assistant\n{{content}}\n')
        ]
    ),
    function_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<extra_id_1>Assistant\n{{content}}\n')
        ]
    ),
    observation_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<extra_id_1>Tool\n{{content}}\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<extra_id_0>System\n{{content}}\n\n')
        ]
    ),
    separator=TemplateComponent(type='string', content='\n'),
    remove_last_sep=True,
    special_stopper=TemplateComponent(type='token', content='eos_token')
)