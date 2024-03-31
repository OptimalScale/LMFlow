from __future__ import absolute_import
import unittest

from transformers import AutoTokenizer

from src.lmflow.utils.conversation_template import ConversationTemplate

MODEL_PATH = 'meta-llama/Llama-2-7b-hf'


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
        [0, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962],
        [6324, 29991, 0]
    )
]

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
        [0, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962], 
        [6324, 29991, 0]
    ), 
    (
        [0, 518, 25580, 29962, 1128, 526, 366, 29973, 518, 29914, 25580, 29962], 
        [306, 29915, 29885, 1781, 29892, 3969, 29991, 0]
    )
]


class TestConversationTemplate(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.conversation_template = ConversationTemplate()

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