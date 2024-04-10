from __future__ import absolute_import
import unittest

from transformers import AutoTokenizer

from lmflow.utils.conversation_template import EmptyConversationTemplate, Llama2ConversationTemplate, QwenConversationTemplate


conversation_singleturn = {
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

conversation_singleturn_llama2 = [
    {
        "role": "user",
        "content": "[INST] <<SYS>>\nsysinfo\n<</SYS>>\n\nHello [/INST]"
    },
    {
        "role": "assistant",
        "content": "Hi!"
    }
]

conversation_singleturn_llama2_ids = [
    (
        [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962],
        [6324, 29991, 2]
    )
]

conversation_multiturn = {
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

conversation_multiturn_llama2 = [
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

conversation_multiturn_llama2_ids = [
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
        self.conversation_template = EmptyConversationTemplate()

    def test_encode_conversation_singleturn_llama2(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_singleturn_llama2,
            system=None,
            tools=None
        )
        self.assertEqual(res, conversation_singleturn_llama2_ids)

    def test_encode_conversation_multiturn_llama2(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_multiturn_llama2,
            system=None,
            tools=None
        )
        self.assertEqual(res, conversation_multiturn_llama2_ids)
        
        
class Llama2ConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        MODEL_PATH = 'meta-llama/Llama-2-7b-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.conversation_template = Llama2ConversationTemplate()
        
    def test_encode_conversation_singleturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_singleturn['messages'],
            system=conversation_singleturn['system'],
            tools=None
        )
        self.assertEqual(res, conversation_singleturn_llama2_ids)
        
    def test_encode_conversation_multiturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_multiturn['messages'],
            system=conversation_multiturn['system'],
            tools=None
        )
        self.assertEqual(res, conversation_multiturn_llama2_ids)
        
        
class Qwen1_5ConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        MODEL_PATH = 'Qwen/Qwen1.5-0.5B-Chat'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.conversation_template = QwenConversationTemplate()
        
    def test_encode_conversation_singleturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_singleturn['messages'],
            system=conversation_singleturn['system'],
            tools=None
        )
        print(res)
        for r in res:
            print(self.tokenizer.decode(r[0]))
            print(self.tokenizer.decode(r[1]))
        
    def test_encode_conversation_multiturn(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_multiturn['messages'],
            system=conversation_multiturn['system'],
            tools=None
        )
        print(res)
        for r in res:
            print(self.tokenizer.decode(r[0]))
            print(self.tokenizer.decode(r[1]))
            
        # print('===')
        # print(self.tokenizer.apply_chat_template(
        #     conversation_multiturn['messages'],
        #     tokenize=False,
        #     add_generation_prompt=True
        # ))
        # print('===')


if __name__ == '__main__':
    unittest.main()