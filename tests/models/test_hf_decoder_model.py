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


SAMPLE_TEXT = "Defintion: In this task, we ask you to write an answer to a question that involves events that may be stationary (not changing over time) or transient (changing over time). For example, the sentence \"he was born in the U.S.\" contains a stationary event since it will last forever; however, \"he is hungry\" contains a transient event since it will remain true for a short period of time. Note that a lot of the questions could have more than one correct answer. We only need a single most-likely answer. Please try to keep your \"answer\" as simple as possible. Concise and simple \"answer\" is preferred over those complex and verbose ones. \\n Input: Sentence: It's hail crackled across the comm, and Tara spun to retake her seat at the helm. \nQuestion: Will the hail storm ever end? \\n Output: NA \\n\\n"

SAMPLE_TOKENS = [
        7469, 600, 295, 25, 554, 428, 4876, 11, 356, 1265, 345, 284, 3551, 281, 3280, 284, 257, 1808, 326, 9018, 2995, 326, 743, 307, 31607, 357, 1662, 5609, 625, 640, 8, 393, 32361, 357, 22954, 625, 640, 737, 1114, 1672, 11, 262, 6827, 366, 258, 373, 4642, 287, 262, 471, 13, 50, 526, 4909, 257, 31607, 1785, 1201, 340, 481, 938, 8097, 26, 2158, 11, 366, 258, 318, 14720, 1, 4909, 257, 32361, 1785, 1201, 340, 481, 3520, 2081, 329, 257, 1790, 2278, 286, 640, 13, 5740, 326, 257, 1256, 286, 262, 2683, 714, 423, 517, 621, 530, 3376, 3280, 13, 775, 691, 761, 257, 2060, 749, 12, 40798, 3280, 13, 4222, 1949, 284, 1394, 534, 366, 41484, 1, 355, 2829, 355, 1744, 13, 13223, 786, 290, 2829, 366, 41484, 1, 318, 9871, 625, 883, 3716, 290, 15942, 577, 3392, 13, 3467, 77, 23412, 25, 11352, 594, 25, 632, 338, 32405, 8469, 992, 1973, 262, 725, 11, 290, 37723, 26843, 284, 41754, 607, 5852, 379, 262, 18030, 13, 220, 198, 24361, 25, 2561, 262, 32405, 6388, 1683, 886, 30, 3467, 77, 25235, 25, 11746, 3467, 77, 59, 77
]

SAMPLE_ATTENTION_MASKS = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

test_encode_input = "Question: Which of the following is not true for myelinated nerve fibers: (A) Impulse through myelinated fibers is slower than non-myelinated fibers (B) Membrane currents are generated at nodes of Ranvier (C) Saltatory conduction of impulses is seen (D) Local anesthesia is effective only when the nerve is not covered by myelin sheath."
test_encode_output = [24361, 25, 9022, 286, 262, 1708, 318, 407, 2081, 329, 616, 417, 3898, 16384, 26742, 25, 357, 32, 8, 9855, 9615, 832, 616, 417, 3898, 26742, 318, 13611, 621, 1729, 12, 1820, 417, 3898, 26742, 357, 33, 8, 4942, 1671, 1531, 28629, 389, 7560, 379, 13760, 286, 23075, 49663, 357, 34, 8, 13754, 2870, 369, 11124, 286, 37505, 318, 1775, 357, 35, 8, 10714, 49592, 318, 4050, 691, 618, 262, 16384, 318, 407, 5017, 416, 616, 27176, 673, 776, 13]
test_decode_input = [24361, 25, 9022, 286, 262, 1708, 318, 407, 2081, 329, 616, 417, 3898, 16384, 26742, 25, 357, 32, 8, 9855, 9615, 832, 616, 417, 3898, 26742, 318, 13611, 621, 1729, 12, 1820, 417, 3898, 26742, 357, 33, 8, 4942, 1671, 1531, 28629, 389, 7560, 379, 13760, 286, 23075, 49663, 357, 34, 8, 13754, 2870, 369, 11124, 286, 37505, 318, 1775, 357, 35, 8, 10714, 49592, 318, 4050, 691, 618, 262, 16384, 318, 407, 5017, 416, 616, 27176, 673, 776, 13]
test_decode_output = "Question: Which of the following is not true for myelinated nerve fibers: (A) Impulse through myelinated fibers is slower than non-myelinated fibers (B) Membrane currents are generated at nodes of Ranvier (C) Saltatory conduction of impulses is seen (D) Local anesthesia is effective only when the nerve is not covered by myelin sheath."

test_inference_output = "The following is a list of the most common causes of myelinated nerve fibers."

class HFDecoderModelTest(unittest.TestCase):

    def _test_tokenize(
        self,
        model_name,
        groundtruth_dataset,
        groundtruth_tokenized_dataset,
    ):
        data_args = DatasetArguments(dataset_path=None)
        dataset = Dataset(data_args, backend="huggingface")
        dataset = dataset.from_dict(groundtruth_dataset)

        self.assertEqual(dataset.to_dict(), groundtruth_dataset)

        model_args = ModelArguments(model_name_or_path=model_name)
        model = HFDecoderModel(model_args)

        tokenized_dataset = model.tokenize(dataset)

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
