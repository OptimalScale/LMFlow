from __future__ import absolute_import
import unittest

from transformers import AutoTokenizer

from lmflow.utils.conversation_template import ConversationTemplate, Llama2ConversationTemplate
from lmflow.utils.conversation_formatter import StringFormatter, EmptyFormatter, TemplateComponent

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

conversation_singleturn_llama2_no_template = {
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

conversation_singleturn_llama2_ids = [
    (
        [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962],
        [6324, 29991, 2]
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

conversation_multiturn_llama2_no_template = {
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


class Llama2ConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.user_foramtter = StringFormatter(
            template=[
                TemplateComponent(type='token', content='bos_token'),
                TemplateComponent(type='string', content='[INST] {{content}} [/INST]')
            ]
        )
        self.system_formatter = StringFormatter(
            template=[
                TemplateComponent(type='string', content='<<SYS>>\n{{content}}\n<</SYS>>\n\n')
            ]
        )
        self.assistant_formatter = StringFormatter(
            template=[
                TemplateComponent(type='string', content='{{content}}'),
                TemplateComponent(type='token', content='eos_token')
            ]
        )
        self.tools_formatter = EmptyFormatter(
            template=[
                TemplateComponent(type='string', content='')
            ]
        ) # TODO
        self.conversation_template = Llama2ConversationTemplate(
            user_formatter=self.user_foramtter,
            assistant_formatter=self.assistant_formatter,
            system_formatter=self.system_formatter,
            tools_formatter=self.tools_formatter
        )
        
    def test_encode_conversation_singleturn_llama2_no_template(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_singleturn_llama2_no_template['messages'],
            system=conversation_singleturn_llama2_no_template['system'],
            tools=None
        )
        self.assertEqual(res, conversation_singleturn_llama2_ids)
        
    def test_encode_conversation_multiturn_llama2_no_template(self):
        res = self.conversation_template.encode_conversation(
            tokenizer=self.tokenizer,
            messages=conversation_multiturn_llama2_no_template['messages'],
            system=conversation_multiturn_llama2_no_template['system'],
            tools=None
        )
        self.assertEqual(res, conversation_multiturn_llama2_ids)
        
        
class EmptyConversationTemplateTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        self.user_foramtter = StringFormatter(
            template=[
                TemplateComponent(type='token', content='bos_token'),
                TemplateComponent(type='string', content='{{content}}')
            ]
        )
        self.assistant_formatter = StringFormatter(
            template=[
                TemplateComponent(type='string', content='{{content}}'),
                TemplateComponent(type='token', content='eos_token')
            ]
        )
        self.conversation_template = ConversationTemplate(
            user_formatter=self.user_foramtter,
            assistant_formatter=self.assistant_formatter,
            system_formatter=None,
            tools_formatter=None
        )

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


if __name__ == '__main__':
    unittest.main()