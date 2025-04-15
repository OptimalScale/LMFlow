#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import os
import sys
import copy

from transformers import (
    HfArgumentParser
)

from lmflow.datasets import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.dpov2_aligner import DPOv2Aligner
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    DPOv2AlignerArguments,
)
from lmflow.utils.common import remove_dataclass_attr_prefix, create_copied_dataclass


logger = logging.getLogger(__name__)


ReferenceModelArguments: ModelArguments = create_copied_dataclass(
    original_dataclass=ModelArguments, 
    field_prefix="reference_",
    class_prefix="Reference"
)


def main():
    # Parses arguments
    parser = HfArgumentParser((
        ModelArguments, 
        ReferenceModelArguments,
        DatasetArguments,
        DPOv2AlignerArguments,
    ))
    target_model_args, ref_model_args, data_args, aligner_args = parser.parse_args_into_dataclasses()
        
    ref_model_args_dict = remove_dataclass_attr_prefix(ref_model_args, "reference_")
    ref_model_args = ModelArguments(**ref_model_args_dict)

    target_model = HFDecoderModel(target_model_args)
    ref_model = HFDecoderModel(ref_model_args)
    train_dataset = Dataset(data_args)
    eval_dataset = copy.deepcopy(train_dataset.sample(
        n=100, 
        seed=aligner_args.random_seed
    ))
    
    aligner = DPOv2Aligner(
        model_args=target_model_args,
        data_args=train_dataset.data_args,
        aligner_args=aligner_args,
        ref_model_args=ref_model.model_args,
    )
    aligner.align(
        model=target_model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    

if __name__ == "__main__":
    main()