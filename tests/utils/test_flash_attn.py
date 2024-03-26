from __future__ import absolute_import
import unittest
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from pathlib import Path

import torch
from transformers.deepspeed import HfDeepSpeedConfig

from lmflow.args import DatasetArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.utils.constants import (
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
)


model_name = 'meta-llama/Llama-2-7b-hf'


data_args = DatasetArguments(dataset_path=None)
dataset = Dataset(data_args, backend="huggingface")
# dataset = dataset.from_dict(groundtruth_dataset)


model_args = ModelArguments(
    model_name_or_path=model_name, 
    use_flash_attention=True
)

model = AutoModel.get_model(
    model_args,
    tune_strategy='none',
    use_accelerator=True,
)

inputs = model.encode(['''hi'''], return_tensors='pt')

res = model.inference(
	inputs=inputs['input_ids'].to("cuda:0"), 
	attention_mask=inputs['attention_mask'].to("cuda:0"),
	use_accelerator=True,
	max_new_tokens=64,
	do_sample=True
)

print(torch.cuda.memory_summary(device=0, abbreviated=False))
print(torch.cuda.memory_allocated(device=0))
print(model.decode(res))