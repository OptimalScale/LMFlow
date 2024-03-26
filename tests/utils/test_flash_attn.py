from __future__ import absolute_import
import unittest
import json
import os
from pathlib import Path

import torch
from transformers.deepspeed import HfDeepSpeedConfig

from lmflow.args import DatasetArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.utils.constants import (
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
)


model_name = '/home/yizhenjia/projs/llama/llama-2-7b-chat-hf'


class FlashAttentionTest(unittest.TestCase):
    pass


data_args = DatasetArguments(dataset_path=None)
dataset = Dataset(data_args, backend="huggingface")
# dataset = dataset.from_dict(groundtruth_dataset)


model_args = ModelArguments(
    model_name_or_path=model_name, 
    use_flash_attention=True
)
model = HFDecoderModel(model_args, use_accelerator=True)
model.get_backend_model().to('cuda:6')
model.get_backend_model().device

inputs = model.encode(['hi'], return_tensors='pt')

model.inference(
    inputs=inputs['input_ids'], 
    attention_mask=inputs['attention_mask'],
    use_accelerator=True)
model.get_backend_model().device
print(torch.cuda.memory_summary(device=5, abbreviated=False))
torch.cuda.memory_allocated(device=5)

