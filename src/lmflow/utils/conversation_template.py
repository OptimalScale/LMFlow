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
    special_starter: Optional[TemplateComponent] = None
    special_stopper: Optional[TemplateComponent] = None
    
    def __post_init__(self):
        if self.separator:
            if self.separator.type not in ['string', 'token']:
                raise NotImplementedError(f"Component type {self.separator.type} cannot be used as a separator.")
            
        if self.special_starter:
            if self.special_starter.type not in ['string', 'token', 'token_id']:
                raise NotImplementedError(f"Component type {self.special_starter.type} cannot be used as a special starter.")
    
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
            
        if self.special_starter:
            # For models that has ONLY ONE bos token at the beginning of 
            # a conversation session (not a conversation pair), user can
            # specify a special starter to add that starter to the very
            # beginning of the conversation session. 
            # eg:
            #   llama-2: <s> and </s> at every pair of conversation 
            #   v.s.
            #   llama-3: <|begin_of_text|> only at the beginning of a session
            encoded_pairs = self.add_special_starter(encoded_pairs, tokenizer)
            
        if self.special_stopper:
            encoded_pairs = self.add_special_stopper(encoded_pairs, tokenizer)
        
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
                    encoded_ids += self._ensure_id_list(tokenizer.convert_tokens_to_ids(component.content))
            elif component.type == 'token_id':
                encoded_ids += self._ensure_id_list(component.content)
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
            separator_ids = self._ensure_id_list(tokenizer.convert_tokens_to_ids(self.separator.content))
        else:
            raise ValueError(f"Component type {self.separator.type} cannot be used as a separator.")
        
        if len(separator_ids) > len(last_assistant_msg):
            raise ValueError("Separator is longer than the last assistant message, please check.")
        
        if last_assistant_msg[-len(separator_ids):] == separator_ids:
            last_assistant_msg = last_assistant_msg[:-len(separator_ids)]
            
        encoded_pairs[-1] = (encoded_pairs[-1][0], last_assistant_msg)
        
        return encoded_pairs
    
    def add_special_starter(
        self,
        encoded_pairs: Sequence[Tuple[List[int], List[int]]],
        tokenizer: PreTrainedTokenizer
    ) -> Sequence[Tuple[List[int], List[int]]]:
        if self.special_starter.type == 'string':
            special_starter_ids = tokenizer.encode(self.special_starter.content, add_special_tokens=False)
        elif self.special_starter.type == 'token':
            if self.special_starter.content == 'bos_token':
                special_starter_ids = [tokenizer.bos_token_id]
            elif self.special_starter.content == 'eos_token':
                special_starter_ids = [tokenizer.eos_token_id]
            else:
                special_starter_ids = self._ensure_id_list(tokenizer.convert_tokens_to_ids(self.special_starter.content))
        elif self.special_starter.type == 'token_id':
            special_starter_ids = self._ensure_id_list(self.special_starter.content)
        else:
            raise ValueError(f"Component type {self.special_starter.type} cannot be used as a special starter.")
        
        encoded_pairs[0] = (special_starter_ids + encoded_pairs[0][0], encoded_pairs[0][1])
        
        return encoded_pairs
    
    def add_special_stopper(
        self,
        encoded_pairs: Sequence[Tuple[List[int], List[int]]],
        tokenizer: PreTrainedTokenizer
    ) -> Sequence[Tuple[List[int], List[int]]]:
        if self.special_stopper.type == 'string':
            special_stopper_ids = tokenizer.encode(self.special_stopper.content, add_special_tokens=False)
        elif self.special_stopper.type == 'token':
            if self.special_stopper.content == 'bos_token':
                special_stopper_ids = [tokenizer.bos_token_id]
            elif self.special_stopper.content == 'eos_token':
                special_stopper_ids = [tokenizer.eos_token_id]
            else:
                special_stopper_ids = self._ensure_id_list(tokenizer.convert_tokens_to_ids(self.special_stopper.content))
        elif self.special_stopper.type == 'token_id':
            special_stopper_ids = self._ensure_id_list(self.special_stopper.content)
        else:
            raise ValueError(f"Component type {self.special_stopper.type} cannot be used as a special stopper.")
        
        encoded_pairs[-1] = (encoded_pairs[-1][0], encoded_pairs[-1][1] + special_stopper_ids)
        
        return encoded_pairs
    
    def _ensure_id_list(self, obj: Union[int, List[int]]) -> List[int]:
        '''Make sure the object is a list of integers. Useful for handling token ids.
        '''
        if isinstance(obj, int):
            return [obj]
        elif isinstance(obj, list):
            return obj
        else:
            raise ValueError(f"Object type {type(obj)} is not supported yet.")
    

@dataclass
class ChatMLConversationTemplate(ConversationTemplate):
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
    

@dataclass
class DeepSeekConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='User: {{content}}\n\n')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='Assistant: {{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    )
    system_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}\n\n')
        ]
    )
    special_starter: TemplateComponent = TemplateComponent(type='token', content='bos_token')


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
class Llama3ConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>assistant<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    )
    system_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>')
        ]
    )
    special_starter: TemplateComponent = TemplateComponent(type='token', content='<|begin_of_text|>')


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
class Phi3ConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|user|>\n{{content}}<|end|>\n')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|assistant|>\n{{content}}<|end|>\n')
        ]
    )
    system_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|system|>\n{{content}}<|end|>\n')
        ]
    )
    special_starter: TemplateComponent = TemplateComponent(type='token', content='bos_token')
    special_stopper: TemplateComponent = TemplateComponent(type='token', content='eos_token')
    
    
@dataclass
class Qwen2ConversationTemplate(ChatMLConversationTemplate):
    separator: TemplateComponent = TemplateComponent(type='string', content='\n')