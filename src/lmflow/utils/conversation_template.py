import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from transformers import PreTrainedTokenizer

from .constants import CONVERSATION_ROLE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class ConversationTemplate:
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
        TODO: Support for different models.
        Data example: 
        ```json
        {
            "conversation_id": 2,
            "system": "sysinfo1",
            "tools": ["tool_1_desc"],
            "messages": [
                {
                    "role": "human",
                    "content": "I have a M via API so"
                },
                {
                    "role": "gpt",
                    "content": "To"
                }
            ]
        }
        ```
        '''        
        encoded_pairs = self.__encode(tokenizer, messages, system, tools, **kwargs)
        
        return encoded_pairs
        
    def __encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:
        # TODO: truncation according to model max length
        # TODO: make sure the last few tokens are "learnable", not masked with token_id = -100.
        bos = [] if kwargs.get("disable_conversation_bos_token", False) else [tokenizer.bos_token_id]
        eos = [] if kwargs.get("disable_conversation_eos_token", False) else [tokenizer.eos_token_id]
        res_all = []
        
        for i in range(0, len(messages), 2):
            user_message = messages[i]
            system_message = messages[i + 1]
            
            user_input = user_message["content"]
            system_input = system_message["content"]
            
            user_encoded = tokenizer.encode(user_input, add_special_tokens=False)
            system_encoded = tokenizer.encode(system_input, add_special_tokens=False)
            
            res_all.append((
                bos + user_encoded, 
                system_encoded + eos
            ))
            
        return res_all
            
            
@dataclass
class Llama2ConversationTemplate(ConversationTemplate):
    def __encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs
    ) -> Sequence[Tuple[List[int], List[int]]]:
        '''The system info is included in the first round of user input for Llama2.
        '''
        pass