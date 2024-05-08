import logging
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from .base import StringFormatter, TemplateComponent, ConversationTemplate


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