from __future__ import absolute_import
import unittest

from transformers import AutoTokenizer

from lmflow.utils.conversation_template import PRESET_TEMPLATES


CONVERSATION_SINGLETURN = {
    "system": "sysinfo",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hi!"
        }
    ]
}

CONVERSATION_SINGLETURN_LLAMA2 = [
    {
        "role": "user",
        "content": "[INST] <<SYS>>\nsysinfo\n<</SYS>>\n\nHello [/INST]"
    },
    {
        "role": "assistant",
        "content": "Hi!"
    }
]

CONVERSATION_SINGLETURN_LLAMA2_IDS = [
    (
        [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962],
        [6324, 29991, 2]
    )
]

CONVERSATION_MULTITURN = {
    "system": "sysinfo",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hi!"
        },
        {
            "role": "user",
            "content": "How are you?"
        },
        {
            "role": "assistant",
            "content": "I'm good, thanks!"
        }
    ]
}

CONVERSATION_MULTITURN_LLAMA2 = [
    {
        "role": "user",
        "content": "[INST] <<SYS>>\nsysinfo\n<</SYS>>\n\nHello [/INST]"
    },
    {
        "role": "assistant",
        "content": "Hi!"
    },
    {
        "role": "user",
        "content": "[INST] How are you? [/INST]"
    },
    {
        "role": "assistant",
        "content": "I'm good, thanks!"
    }
]

CONVERSATION_MULTITURN_LLAMA2_IDS = [
    (
        [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962], 
        [6324, 29991, 2]
    ), 
    (
        [1, 518, 25580, 29962, 1128, 526, 366, 29973, 518, 29914, 25580, 29962], 
        [306, 29915, 29885, 1781, 29892, 3969, 29991, 2]
    )
]


class EmptyConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        MODEL_PATH = 'meta-llama/Llama-2-7b-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.conversation_template = PRESET_TEMPLATES['empty']

    def test_encode_conversation_singleturn_llama2(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=CONVERSATION_SINGLETURN_LLAMA2,
            system=None,
            tools=None
        )
        self.assertEqual(res, CONVERSATION_SINGLETURN_LLAMA2_IDS)

    def test_encode_conversation_multiturn_llama2(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=CONVERSATION_MULTITURN_LLAMA2,
            system=None,
            tools=None
        )
        self.assertEqual(res, CONVERSATION_MULTITURN_LLAMA2_IDS)
        
        
class Llama2ConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        MODEL_PATH = 'meta-llama/Llama-2-7b-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.conversation_template = PRESET_TEMPLATES['llama2']
        
    def test_encode_conversation_singleturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=CONVERSATION_SINGLETURN['messages'],
            system=CONVERSATION_SINGLETURN['system'],
            tools=None
        )
        self.assertEqual(res, CONVERSATION_SINGLETURN_LLAMA2_IDS)
        
    def test_encode_conversation_multiturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=CONVERSATION_MULTITURN['messages'],
            system=CONVERSATION_MULTITURN['system'],
            tools=None
        )
        self.assertEqual(res, CONVERSATION_MULTITURN_LLAMA2_IDS)
        
        
class Qwen2ConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        MODEL_PATH = 'Qwen/Qwen1.5-0.5B-Chat'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.conversation_template = PRESET_TEMPLATES['qwen2']
        
    def test_encode_conversation_singleturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=CONVERSATION_SINGLETURN['messages'],
            system=CONVERSATION_SINGLETURN['system'],
            tools=None
        )
        print(res)
        
    def test_encode_conversation_multiturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=CONVERSATION_MULTITURN['messages'],
            system=CONVERSATION_MULTITURN['system'],
            tools=None
        )
        print(res)
            
        print('===')
        print(self.tokenizer.apply_chat_template(
            CONVERSATION_MULTITURN['messages'],
            tokenize=True,
            add_generation_prompt=False
        ))
        print('===')
        print(self.tokenizer.apply_chat_template(
            CONVERSATION_MULTITURN['messages'],
            tokenize=False,
            add_generation_prompt=False
        ))
        print('===')


if __name__ == '__main__':
    unittest.main()