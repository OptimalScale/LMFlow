#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple script to merge lora weights.
"""

import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments




@dataclass
class OtherArguments:
    save_dir: Optional[str] = field(
        default="./save_merged_lora",
        metadata={
            "help": "Dir to save merged lora"
        },
    )



def main():

    parser = HfArgumentParser((
        ModelArguments,
        OtherArguments,
    ))
    model_args, other_args = (
        parser.parse_args_into_dataclasses()
    )

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        device="cpu",
    )

    model.merge_lora_weights()

    if not os.path.exists(other_args.save_dir):
        os.mkdir(other_args.save_dir)

    model.save(other_args.save_dir,save_full_model = True)







if __name__ == "__main__":
    main()
