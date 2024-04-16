import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from transformers import PreTrainedTokenizer

from .conversation_formatter import Formatter, TemplateComponent, StringFormatter, EmptyFormatter


logger = logging.getLogger(__name__)


@dataclass
class ConversationTemplate:
    user_formatter: Formatter
    assistant_formatter: Formatter
    system_formatter: Optional[Formatter] = None
    tools_formatter: Optional[Formatter] = None
    separator: Optional[TemplateComponent] = None
    
    def encode_conversation(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
        remove_last_sep: bool = False,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r'''
        Messages here should be guaranteed to be in pairs, with the first message being the user message and the second message being the system message.
        Data example: 
        ```json
        {
            "conversation_id": 2,
            "system": "sysinfo1",
            "tools": ["tool_1_desc"],
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
                {
                    "role": "assistant",
                    "content": "Hello!"
                }
            ]
        }
        ```
        '''
        assert isinstance(messages, list), "Messages must be a list."
        
        if tools:
            logger.warning("Tools are not supported yet. Please include tools in the system message manually.")
        
        if system:
            if system.replace(" ",""):
                if not self.system_formatter:
                    raise ValueError("Your dataset contains system message but no system formatter is provided. "
                                     "Consider either providing a system formatter or removing system prompt from your dataset.")
            else:
                system = None
        
        encoded_pairs = self._encode(tokenizer, messages, system, tools, **kwargs)
        
        if self.separator and remove_last_sep:
            # For models that require a separator between messages, 
            # user can include the seperator at the end of each template
            # and specify the separator. Auto formatting will remove the 
            # last separator once user specifies this option.
            encoded_pairs = self.remove_last_separator(encoded_pairs, tokenizer)
        
        return encoded_pairs
        
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
            assistant_formatted = self.assistant_formatter.format(content=assistant_message["content"])
            
            user_encoded = self._encode_template(user_formatted, tokenizer)
            assistant_encoded = self._encode_template(assistant_formatted, tokenizer)
            
            res_all.append((
                system_encoded + user_encoded if i == 0 else user_encoded, 
                assistant_encoded
            ))
            
        return res_all
    
    def _encode_template(
        self, 
        template: List[TemplateComponent],
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ) -> List[int]:
        """Encode template components into token ids.

        Parameters
        ----------
        template : List[TemplateComponent]
            Formatted template components.
        tokenizer : PreTrainedTokenizer
            Tokenizer to convert tokens into token ids.

        Returns
        -------
        List[int]
            Encoded token ids.
        """
        encoded_ids = []
        for component in template:
            if component.type == 'string':
                if len(component.content) == 0:
                    logger.warning("Empty string component found in the template.")
                    continue
                else:
                    encoded_ids += tokenizer.encode(component.content, add_special_tokens=False)
            elif component.type == 'token':
                if component.content == 'bos_token':
                    encoded_ids += [tokenizer.bos_token_id]
                elif component.content == 'eos_token':
                    encoded_ids += [tokenizer.eos_token_id]
                else:
                    encoded_ids += [tokenizer.convert_tokens_to_ids(component.content)]
            elif component.type == 'token_id':
                encoded_ids += [component.content] if isinstance(component.content, int) else component.content
            else:
                raise NotImplementedError(f"Component type {component.type} is not supported yet.")
        return encoded_ids
    
    def remove_last_separator(
        self, 
        encoded_pairs: Sequence[Tuple[List[int], List[int]]],
        tokenizer: PreTrainedTokenizer
    ) -> Sequence[Tuple[List[int], List[int]]]:
        last_assistant_msg = encoded_pairs[-1][1]
        if self.separator.type == 'string':
            separator_ids = tokenizer.encode(self.separator.content, add_special_tokens=False)
        elif self.separator.type == 'token':
            separator_ids = [tokenizer.convert_tokens_to_ids(self.separator.content)]
        else:
            raise NotImplementedError(f"Component type {self.separator.type} cannot be used as a separator.")
        
        if len(separator_ids) > len(last_assistant_msg):
            raise ValueError("Separator is longer than the last assistant message, please check.")
        
        if last_assistant_msg[-len(separator_ids):] == separator_ids:
            last_assistant_msg = last_assistant_msg[:-len(separator_ids)]
            
        encoded_pairs[-1] = (encoded_pairs[-1][0], last_assistant_msg)
        
        return encoded_pairs
    
            
@dataclass
class EmptyConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='{{content}}')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    )
    

@dataclass
class EmptyConversationTemplateWithoutSpecialTokens(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}')
        ]
    )
    
            
@dataclass
class Llama2ConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='[INST] {{content}} [/INST]')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    )
    system_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<<SYS>>\n{{content}}\n<</SYS>>\n\n')
        ]
    )
    
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
    
    
@dataclass
class Qwen2ConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>user\n{{content}}<|im_end|>\n')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    )
    system_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>system\n{{content}}<|im_end|>\n')
        ]
    )
    separator: TemplateComponent = TemplateComponent(type='string', content='\n')