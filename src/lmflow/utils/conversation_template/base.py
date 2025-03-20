#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple
import logging

from transformers import PreTrainedTokenizer

from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class TemplateComponent:
    """The minimal unit of a template, which can be a token, a string, or a list of tools.

    Parameters
    ----------
    type : Literal['token', 'token_id', 'string', 'tools']
        - Type of the component.  
        
        - When the component is a token or a string, the content should be `string`. 
        The difference between the two is that token will be converted to token ids 
        by the tokenizer.convert_tokens_to_ids() method, while string will be directly 
        encoded by the tokenizer.encode() method. Specially, since the bos token and eos
        token are frequently used across different templates, we provide the convenience
        to use `'bos_token'` and `'eos_token'` to represent the actual bos and eos tokens when
        `type` of the `TemplateComponent` is `token`. For example:  
        
        ```python
        TemplateComponent(type='token', content='bos_token')
        ```
        
        After encoding, the content will be replaced by the actual token id of the bos token.
        Please do remember that if you set the `type` to `string`, the tokenizer will try to 
        encode the string 'bos_token' instead of providing the actual bos token.
        
        - When the component is token_id, the content should be `int` or `List[int]`, and 
        will be directly appended to the encoded token ids.
        
        - Tools are not supported yet.
        
    content : Union[str, int, List[str], List[int]]
        Content of the component.

    """
    type: Literal['token', 'token_id', 'string', 'tools']
    content: Union[str, int, List[str], List[int]]
    mask: Optional[bool] = True # for token specific masking, work in progress
    
    def __post_init__(self):
        assert self.content, "Content of the component cannot be empty."
        
        if self.type == 'tools':
            assert isinstance(self.content, list), (
                f"Content of tools component must be a list, got {type(self.content)}")
        elif self.type in ['token', 'string']:
            assert isinstance(self.content, str), (
                f"Content of string/token component must be a string, got {type(self.content)}")
        elif self.type == 'token_id':
            assert isinstance(self.content, int) or all(isinstance(token_id, int) for token_id in self.content), (
                f"Content of token_id component must be an integer or a list of integers.")
        else:
            raise ValueError(f"The type of the component must be either "
                             f"'token', 'string' or 'tools', got {self.type}")
            
    def __repr__(self) -> str:
        return f"TemplateComponent(type={self.type}, content={self.content})".replace("\n", "\\n")
    
    def __str__(self) -> str:
        return f"{self.content}".replace("\n", "\\n")


@dataclass
class Formatter(ABC):
    template: List[TemplateComponent] = field(default_factory=list)
    
    @abstractmethod
    def format(self, **kwargs) -> List[TemplateComponent]: ...
    
    def has_placeholder(self):
        flag = False
        for component in self.template:
            if component.type == 'string':
                if re.search(r"{{(.*?)}}", component.content):
                    flag = True
                    break
        return flag


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        if self.has_placeholder():
            raise ValueError("Empty formatter should not have placeholders.")
    
    def format(self, **kwargs) -> list:
        """Empty formatter for when no formatting is needed.
        This is useful when user has already applied formatting to the dataset.

        Returns
        -------
        list
            Original template.
        """
        return self.template
    

@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        if not self.has_placeholder():
            raise ValueError("String formatter should have placeholders.")
    
    def format(self, **kwargs) -> list:
        """Format the string components with the provided keyword arguments. 
        Mostly used for formatting system prompt, user and assistant messages.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing values to replace in the template components.

        Returns
        -------
        list
            Formatted template.
        """
        formatted_template = []
        for component in self.template:
            if component.type == 'string':
                for key, value in kwargs.items():
                    templated = component.content.replace("{{" + key + "}}", value)
                    if len(templated) == 0:
                        logger.warning("Found empty string after formatting, adding a space instead. "
                                       "If this is not intended, please check the dataset.")
                        templated = " "
                    formatted_template.append(TemplateComponent(type='string', content=templated))
            else:
                formatted_template.append(component)
                
        logger.debug(formatted_template)
        return formatted_template


@dataclass
class ConversationTemplate:
    user_formatter: Formatter
    assistant_formatter: Formatter
    function_formatter: Optional[Formatter] = None
    observation_formatter: Optional[Formatter] = None
    system_formatter: Optional[Formatter] = None
    force_system: bool = False
    tools_formatter: Optional[Formatter] = None
    separator: Optional[TemplateComponent] = None
    remove_last_sep: bool = False
    special_starter: Optional[TemplateComponent] = None
    special_stopper: Optional[TemplateComponent] = None
    template_name: Optional[str] = None
    system_default: Optional[str] = None
    
    def __post_init__(self):
        if self.separator:
            if self.separator.type not in ['string', 'token', 'token_id']:
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
                system = self.system_default if self.system_default else None
        
        encoded_pairs = self._encode(tokenizer, messages, system, tools, **kwargs)
        encoded_pairs = self.post_process_pairs(encoded_pairs=encoded_pairs, tokenizer=tokenizer)
        
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
        
        if system:
            system_formatted = self.system_formatter.format(content=system)
        else:
            system_formatted = self.system_formatter.format(content='') if self.force_system else []
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
    
    def post_process_pairs(self, encoded_pairs, tokenizer):
        if self.separator and self.remove_last_sep:
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
        elif self.separator.type == 'token_id':
            separator_ids = self._ensure_id_list(self.separator.content)
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
class ConversationTemplateForTool(ConversationTemplate):
    def encode_conversation(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
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
        
        tools = self._handle_tools(tools)
        
        if system is None:
            system = ""
        else:
            if system.replace(" ",""):
                if not self.system_formatter:
                    raise ValueError("Your dataset contains system message but no system formatter is provided. "
                                     "Consider either providing a system formatter or removing system prompt from your dataset.")
            else:
                system = self.system_default if self.system_default else ""
        encoded_pairs = self._encode(tokenizer, messages, system, tools, **kwargs)
        encoded_pairs = self.post_process_pairs(encoded_pairs=encoded_pairs, tokenizer=tokenizer)
        
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
        # Concatenate the system and tools strings
        system = system + tools
        if system:
            system_formatted = self.system_formatter.format(content=system)
        else:
            system_formatted = self.system_formatter.format(content='') if self.force_system else []
        system_encoded = self._encode_template(system_formatted, tokenizer)
        ls_for_save = []
        for i in range(0, len(messages), 1):
            if messages[i]['role'] == CONVERSATION_ROLE_NAMES['user']:
                user_message = messages[i]
                user_formatted = self.user_formatter.format(content=user_message["content"])
                user_encoded = self._encode_template(user_formatted, tokenizer)
                if i == 0:
                    user_encoded = system_encoded + user_encoded
                ls_for_save.append(user_encoded)
            elif messages[i]['role'] == CONVERSATION_ROLE_NAMES['function']:
                function_message = messages[i]
                function_formatted = self.function_formatter.format(content=function_message['content'])
                function_encoded = self._encode_template(function_formatted, tokenizer)
                ls_for_save.append(function_encoded)
            elif messages[i]['role'] == CONVERSATION_ROLE_NAMES['observation']:
                observation_message = messages[i]
                observation_formatted = self.observation_formatter.format(content=observation_message['content'])
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
    
    def _handle_tools(self, tools: Optional[List[str]]) -> str:
        tools_out = ','.join(tools) if tools is not None else ''
        return tools_out


EMPTY_TEMPLATE = ConversationTemplate(
    template_name='empty',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='{{content}}')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}'),
            TemplateComponent(type='token', content='eos_token')
        ]
    )
)


EMPTY_NO_SPECIAL_TOKENS_TEMPLATE = ConversationTemplate(
    template_name='empty_no_special_tokens',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='{{content}}')
        ]
    )
)