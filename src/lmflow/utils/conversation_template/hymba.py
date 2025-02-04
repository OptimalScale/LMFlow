#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import StringFormatter, TemplateComponent, ConversationTemplateForTool
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple

from transformers import PreTrainedTokenizer

# NOTE: 'contexts' are not used in sft
# {{'<extra_id_0>System'}}
# {% for message in messages %}
#   {% if message['role'] == 'system' %}
#       {{'\n' + message['content'].strip()}}
#       {% if tools %}
#           {{'\n'}}
#       {% endif %}
#   {% endif %}
# {% endfor %}
# {% if tools %}
#   {% for tool in tools %}
#       {{ '\n<tool> ' + tool|tojson + ' </tool>' }}
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
    def _handle_tools(self, tools: Optional[List[str]]) -> str:
        tools_out = ''
        if tools is not None:
            for tool in tools:
                tools_out += "\n<tool> " + tool + " </tool>"
        return tools_out


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
            TemplateComponent(type='string', content='<extra_id_0>System{{content}}\n\n')
        ]
    ),
    separator=TemplateComponent(type='token_id', content=13),
    remove_last_sep=True,
    special_stopper=TemplateComponent(type='token', content='eos_token'),
    force_system=True
)