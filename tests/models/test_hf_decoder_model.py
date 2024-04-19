#!/bin/env/python3
# coding=utf-8
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from __future__ import absolute_import
import unittest
import torch
import json
import os
from pathlib import Path
from transformers.deepspeed import HfDeepSpeedConfig

from lmflow.args import DatasetArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.utils.constants import (
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
)
from lmflow.utils.conversation_template import (
    EmptyConversationTemplate,
    Llama2ConversationTemplate,
    Llama3ConversationTemplate,
    EmptyConversationTemplateWithoutSpecialTokens,
)


SAMPLE_TEXT = "Defintion: In this task, we ask you to write an answer to a question that involves events that may be stationary (not changing over time) or transient (changing over time). For example, the sentence \"he was born in the U.S.\" contains a stationary event since it will last forever; however, \"he is hungry\" contains a transient event since it will remain true for a short period of time. Note that a lot of the questions could have more than one correct answer. We only need a single most-likely answer. Please try to keep your \"answer\" as simple as possible. Concise and simple \"answer\" is preferred over those complex and verbose ones. \\n Input: Sentence: It's hail crackled across the comm, and Tara spun to retake her seat at the helm. \nQuestion: Will the hail storm ever end? \\n Output: NA \\n\\n"

SAMPLE_TOKENS = [
        7469, 600, 295, 25, 554, 428, 4876, 11, 356, 1265, 345, 284, 3551, 281, 3280, 284, 257, 1808, 326, 9018, 2995, 326, 743, 307, 31607, 357, 1662, 5609, 625, 640, 8, 393, 32361, 357, 22954, 625, 640, 737, 1114, 1672, 11, 262, 6827, 366, 258, 373, 4642, 287, 262, 471, 13, 50, 526, 4909, 257, 31607, 1785, 1201, 340, 481, 938, 8097, 26, 2158, 11, 366, 258, 318, 14720, 1, 4909, 257, 32361, 1785, 1201, 340, 481, 3520, 2081, 329, 257, 1790, 2278, 286, 640, 13, 5740, 326, 257, 1256, 286, 262, 2683, 714, 423, 517, 621, 530, 3376, 3280, 13, 775, 691, 761, 257, 2060, 749, 12, 40798, 3280, 13, 4222, 1949, 284, 1394, 534, 366, 41484, 1, 355, 2829, 355, 1744, 13, 13223, 786, 290, 2829, 366, 41484, 1, 318, 9871, 625, 883, 3716, 290, 15942, 577, 3392, 13, 3467, 77, 23412, 25, 11352, 594, 25, 632, 338, 32405, 8469, 992, 1973, 262, 725, 11, 290, 37723, 26843, 284, 41754, 607, 5852, 379, 262, 18030, 13, 220, 198, 24361, 25, 2561, 262, 32405, 6388, 1683, 886, 30, 3467, 77, 25235, 25, 11746, 3467, 77, 59, 77
]

SAMPLE_ATTENTION_MASKS = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

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

CONVERSATION_SINGLETURN_LLAMA2 = {
    "messages": [
        {
            "role": "user",
            "content": "[INST] <<SYS>>\nsysinfo\n<</SYS>>\n\nHello [/INST]"
        },
        {
            "role": "assistant",
            "content": "Hi!"
        }
    ]
}

CONVERSATION_SINGLETURN_LLAMA2_IDS = [
    (
        [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 9675, 3888, 13, 
         29966, 829, 14816, 29903, 6778, 13, 13, 10994, 518, 29914, 25580, 29962],
        [6324, 29991, 2]
    )
]

CONVERSATION_SINGLETURN_LLAMA3_IDS = [
    (
        [128000, 128006, 9125, 128007, 271, 7947, 2801, 128009, 128006, 882, 128007, 271, 9906, 128009],
        [128006, 78191, 128007, 271, 13347, 0, 128009]        
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

CONVERSATION_MULTITURN_LLAMA2 = {
    "messages": [
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
}

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

CONVERSATION_MULTITURN_LLAMA3_IDS = [
    (
        [128000, 128006, 9125, 128007, 271, 7947, 2801, 128009, 128006, 882, 128007, 271, 9906, 128009],
        [128006, 78191, 128007, 271, 13347, 0, 128009]        
    ),
    (
        [128006, 882, 128007, 271, 4438, 527, 499, 30, 128009],
        [128006, 78191, 128007, 271, 40, 2846, 1695, 11, 9523, 0, 128009]
    )
]

test_encode_input = "Question: Which of the following is not true for myelinated nerve fibers: (A) Impulse through myelinated fibers is slower than non-myelinated fibers (B) Membrane currents are generated at nodes of Ranvier (C) Saltatory conduction of impulses is seen (D) Local anesthesia is effective only when the nerve is not covered by myelin sheath."
test_encode_output = [24361, 25, 9022, 286, 262, 1708, 318, 407, 2081, 329, 616, 417, 3898, 16384, 26742, 25, 357, 32, 8, 9855, 9615, 832, 616, 417, 3898, 26742, 318, 13611, 621, 1729, 12, 1820, 417, 3898, 26742, 357, 33, 8, 4942, 1671, 1531, 28629, 389, 7560, 379, 13760, 286, 23075, 49663, 357, 34, 8, 13754, 2870, 369, 11124, 286, 37505, 318, 1775, 357, 35, 8, 10714, 49592, 318, 4050, 691, 618, 262, 16384, 318, 407, 5017, 416, 616, 27176, 673, 776, 13]
test_decode_input = [24361, 25, 9022, 286, 262, 1708, 318, 407, 2081, 329, 616, 417, 3898, 16384, 26742, 25, 357, 32, 8, 9855, 9615, 832, 616, 417, 3898, 26742, 318, 13611, 621, 1729, 12, 1820, 417, 3898, 26742, 357, 33, 8, 4942, 1671, 1531, 28629, 389, 7560, 379, 13760, 286, 23075, 49663, 357, 34, 8, 13754, 2870, 369, 11124, 286, 37505, 318, 1775, 357, 35, 8, 10714, 49592, 318, 4050, 691, 618, 262, 16384, 318, 407, 5017, 416, 616, 27176, 673, 776, 13]
test_decode_output = "Question: Which of the following is not true for myelinated nerve fibers: (A) Impulse through myelinated fibers is slower than non-myelinated fibers (B) Membrane currents are generated at nodes of Ranvier (C) Saltatory conduction of impulses is seen (D) Local anesthesia is effective only when the nerve is not covered by myelin sheath."

test_inference_output = "The following is a list of the most common causes of myelinated nerve fibers."


def make_gt_from_conversation_ids(conversation_ids):
    res = {"input_ids": [], "attention_mask": [], "labels": []}
    for turn_idx, turn_content in enumerate(conversation_ids):
        user_content = turn_content[0]
        assistant_content = turn_content[1]
        res["input_ids"].extend(user_content)
        res["input_ids"].extend(assistant_content)
        res['attention_mask'].extend([1] * len(user_content) + [1] * len(assistant_content))
        res['labels'].extend([-100] * len(user_content))
        res['labels'].extend(assistant_content)
    return res


def make_gt_from_conversation_ids_batch(batched_conversation_ids):
    res = {"input_ids": [], "attention_mask": [], "labels": []}
    for conversation_ids in batched_conversation_ids:
        this_res = make_gt_from_conversation_ids(conversation_ids)
        res["input_ids"].append(this_res["input_ids"])
        res["attention_mask"].append(this_res["attention_mask"])
        res["labels"].append(this_res["labels"])
    return res


class HFDecoderModelTest(unittest.TestCase):

    def _test_tokenize(
        self,
        model_name,
        groundtruth_dataset,
        groundtruth_tokenized_dataset,
        **kwargs
    ):
        data_args = DatasetArguments(dataset_path=None, disable_group_texts=False)
        dataset = Dataset(data_args, backend="huggingface")
        dataset = dataset.from_dict(groundtruth_dataset)

        self.assertEqual(dataset.to_dict(), groundtruth_dataset)

        model_args = ModelArguments(model_name_or_path=model_name)
        model = HFDecoderModel(model_args)

        tokenized_dataset = model.tokenize(dataset, **kwargs)

        self.assertEqual(
            tokenized_dataset.get_backend_dataset().to_dict(),
            groundtruth_tokenized_dataset,
        )

    def test_tokenize_text_only(self):
        text_only_dataset = {
            "type": "text_only",
            "instances": [
                {
                    "text": SAMPLE_TEXT
                },
            ],
        }
        text_only_tokenized_dataset = {
            'input_ids': [SAMPLE_TOKENS],
            'attention_mask': [SAMPLE_ATTENTION_MASKS],
            'labels': [SAMPLE_TOKENS],
        }

        self._test_tokenize(
            model_name="gpt2",
            groundtruth_dataset=text_only_dataset,
            groundtruth_tokenized_dataset=text_only_tokenized_dataset,
        )


    def test_tokenize_text_only_multiple(self):
        text_only_dataset = {
            "type": "text_only",
            "instances": [
                { "text": SAMPLE_TEXT },
                { "text": SAMPLE_TEXT },
            ],
        }
        text_only_tokenized_dataset = {
            'input_ids': [SAMPLE_TOKENS, SAMPLE_TOKENS],
            'attention_mask': [SAMPLE_ATTENTION_MASKS, SAMPLE_ATTENTION_MASKS],
            'labels': [SAMPLE_TOKENS, SAMPLE_TOKENS],
        }

        self._test_tokenize(
            model_name="gpt2",
            groundtruth_dataset=text_only_dataset,
            groundtruth_tokenized_dataset=text_only_tokenized_dataset,
        )


    def test_tokenize_text2text(self):
        text2text_dataset = {
            "type": "text2text",
            "instances": [
                {
                    "input": SAMPLE_TEXT,
                    "output": SAMPLE_TEXT,
                },
            ],
        }
        text2text_tokenized_dataset = {
            'input_ids': [SAMPLE_TOKENS + SAMPLE_TOKENS],
            'attention_mask': [SAMPLE_ATTENTION_MASKS + SAMPLE_ATTENTION_MASKS],
            'labels': [ [-100] * len(SAMPLE_TOKENS) + SAMPLE_TOKENS ],
        }

        self._test_tokenize(
            model_name="gpt2",
            groundtruth_dataset=text2text_dataset,
            groundtruth_tokenized_dataset=text2text_tokenized_dataset,
        )


    def test_tokenize_conversation(self):
        conversation_dataset = {
            "type": "conversation",
            "instances": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": SAMPLE_TEXT
                        },
                        {
                            "role": "assistant",
                            "content": SAMPLE_TEXT
                        }
                    ]
                },
            ],
        }
        conversation_tokenized_dataset = {
            'input_ids': [SAMPLE_TOKENS + SAMPLE_TOKENS],
            'attention_mask': [SAMPLE_ATTENTION_MASKS + SAMPLE_ATTENTION_MASKS],
            'labels': [ [-100] * len(SAMPLE_TOKENS) + SAMPLE_TOKENS ],
        }
        
        self._test_tokenize(
            model_name="gpt2",
            groundtruth_dataset=conversation_dataset,
            groundtruth_tokenized_dataset=conversation_tokenized_dataset,
            conversation_template=EmptyConversationTemplateWithoutSpecialTokens()
        )
        
        self._test_tokenize(
            model_name='meta-llama/Llama-2-7b-hf',
            groundtruth_dataset={"type": "conversation", "instances": [CONVERSATION_SINGLETURN_LLAMA2]},
            groundtruth_tokenized_dataset=make_gt_from_conversation_ids_batch([CONVERSATION_SINGLETURN_LLAMA2_IDS]),
            conversation_template=EmptyConversationTemplate()
        )
        
        self._test_tokenize(
            model_name='meta-llama/Llama-2-7b-hf',
            groundtruth_dataset={"type": "conversation", "instances": [CONVERSATION_SINGLETURN]},
            groundtruth_tokenized_dataset=make_gt_from_conversation_ids_batch([CONVERSATION_SINGLETURN_LLAMA2_IDS]),
            conversation_template=Llama2ConversationTemplate()
        )
        
        self._test_tokenize(
            model_name='meta-llama/Meta-Llama-3-8B-Instruct',
            groundtruth_dataset={"type": "conversation", "instances": [CONVERSATION_SINGLETURN]},
            groundtruth_tokenized_dataset=make_gt_from_conversation_ids_batch([CONVERSATION_SINGLETURN_LLAMA3_IDS]),
            conversation_template=Llama3ConversationTemplate()
        )
        
        
    def test_tokenize_conversation_multiple(self):
        conversation_dataset = {
            "type": "conversation",
            "instances": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": SAMPLE_TEXT
                        },
                        {
                            "role": "assistant",
                            "content": SAMPLE_TEXT
                        }
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": SAMPLE_TEXT
                        },
                        {
                            "role": "assistant",
                            "content": SAMPLE_TEXT
                        }
                    ]
                },
            ],
        }
        conversation_tokenized_dataset = {
            'input_ids': [SAMPLE_TOKENS + SAMPLE_TOKENS, SAMPLE_TOKENS + SAMPLE_TOKENS],
            'attention_mask': [SAMPLE_ATTENTION_MASKS + SAMPLE_ATTENTION_MASKS, SAMPLE_ATTENTION_MASKS + SAMPLE_ATTENTION_MASKS],
            'labels': [ [-100] * len(SAMPLE_TOKENS) + SAMPLE_TOKENS ,  [-100] * len(SAMPLE_TOKENS) + SAMPLE_TOKENS ],
        }
        
        self._test_tokenize(
            model_name="gpt2",
            groundtruth_dataset=conversation_dataset,
            groundtruth_tokenized_dataset=conversation_tokenized_dataset,
            conversation_template=EmptyConversationTemplateWithoutSpecialTokens()
        )
        
        self._test_tokenize(
            model_name='meta-llama/Llama-2-7b-hf',
            groundtruth_dataset={"type": "conversation", "instances": [CONVERSATION_MULTITURN_LLAMA2]},
            groundtruth_tokenized_dataset=make_gt_from_conversation_ids_batch([CONVERSATION_MULTITURN_LLAMA2_IDS]),
            conversation_template=EmptyConversationTemplate()
        )
        
        self._test_tokenize(
            model_name='meta-llama/Llama-2-7b-hf',
            groundtruth_dataset={"type": "conversation", "instances": [CONVERSATION_MULTITURN]},
            groundtruth_tokenized_dataset=make_gt_from_conversation_ids_batch([CONVERSATION_MULTITURN_LLAMA2_IDS]),
            conversation_template=Llama2ConversationTemplate()
        )

        self._test_tokenize(
            model_name='meta-llama/Meta-Llama-3-8B-Instruct',
            groundtruth_dataset={"type": "conversation", "instances": [CONVERSATION_MULTITURN]},
            groundtruth_tokenized_dataset=make_gt_from_conversation_ids_batch([CONVERSATION_MULTITURN_LLAMA3_IDS]),
            conversation_template=Llama3ConversationTemplate()
        )


    def test_encode(self):
        model_name = 'gpt2'
        model_args = ModelArguments(model_name_or_path=model_name)
        model = HFDecoderModel(model_args)
        self.assertEqual(model.encode(test_encode_input), test_encode_output)

        batch_encode_input = [test_encode_input] * 2
        batch_encode_output = [test_encode_output] * 2
        self.assertEqual(model.encode(batch_encode_input)['input_ids'], batch_encode_output)


    def test_decode(self):
        model_name = 'gpt2'
        model_args = ModelArguments(model_name_or_path=model_name)
        model = HFDecoderModel(model_args)
        self.assertEqual(model.decode(test_decode_input), test_decode_output)

        batch_decode_input = [test_decode_input] * 2
        batch_decode_output = [test_decode_output] * 2
        self.assertEqual(model.decode(batch_decode_input), batch_decode_output)


    def test_inference(self):
        ds_config_path = "examples/ds_config.json"
        with open (ds_config_path, "r") as f:
            ds_config = json.load(f)
        dschf = HfDeepSpeedConfig(ds_config)
        model_name = 'gpt2'
        model_args = ModelArguments(
            model_name_or_path=model_name,
            use_ram_optimized_load=False
        )
        model = HFDecoderModel(model_args, tune_strategy='none', ds_config=ds_config)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank) 
        inputs = model.encode(test_encode_input, return_tensors="pt").to(device=self.local_rank)
        outputs = model.inference(inputs,min_length=5, max_length=100,temperature=0.0, do_sample=False)
        text_out = model.decode(outputs[0], skip_special_tokens=True)
        prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
        text_out = text_out[prompt_length:].strip("\n")

        self.assertEqual(text_out, test_inference_output)


if __name__ == "__main__":
    unittest.main()