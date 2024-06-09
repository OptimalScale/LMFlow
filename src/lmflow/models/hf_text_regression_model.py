#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import os
import hashlib
import logging
from pathlib import Path

import torch
import deepspeed
import transformers
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    BitsAndBytesConfig,
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.testing_utils import CaptureLogger

from lmflow.datasets import Dataset
from lmflow.models.interfaces.tunable import Tunable
from lmflow.models.hf_model_mixin import HFModelMixin
from lmflow.models.text_regression_model import TextRegressionModel
from lmflow.tokenization.hf_text_regression_model import tokenize_function
from lmflow.utils.conversation_template import PRESET_TEMPLATES
from lmflow.utils.constants import (
    PAIRED_CONVERSATION_DATASET_DESCRIPTION, 
    CONVERSATION_ROLE_NAMES, 
)


logger = logging.getLogger(__name__)


class HFTextRegressionModel(TextRegressionModel, HFModelMixin, Tunable):
    r"""
    Initializes a HFTextRegressionModel instance.

    Parameters
    ------------

    model_args : 
        Model arguments such as model name, path, revision, etc.

    tune_strategy : str or none,  default="normal".
        A string representing the dataset backend. Defaults to "huggingface".
    
    ds_config :   
        Deepspeed configuations.
    
    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.    
    """

    def __init__(
        self,
        model_args,
        tune_strategy='normal',
        ds_config=None,
        device="gpu",
        use_accelerator=False,
        *args,
        **kwargs
    ):
        """
        Initializes a HFTextRegressionModel instance.
        :param model_args: dictionary with model arguments such as model name, path, revision, etc.
        :param tune_strategy: tuning strategy: normal, none, lora or adapter
        :param ds_config: deepspeed configuration for distributed training
        """
        HFModelMixin.__init__(
            self,
            model_args=model_args,
            do_train=True if tune_strategy == "normal" else False,
            ds_config=ds_config,
            device=device,
            use_accelerator=use_accelerator,
            *args,
            **kwargs
        )

    
    def tokenize(
        self, 
        dataset: Dataset, 
        add_special_tokens=True, 
        *args, 
        **kwargs
    ):
        """
        Tokenize the full dataset.
    
        Parameters
        ------------
        dataset : lmflow.datasets.Dataset.

        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset, without any leading or trailing special
            tokens (normally they are Begin-Of-Sentence or End-Of-Sentence
            tokens).
        """
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        dataset_type = dataset.get_type()
        model_args = self.model_args
        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features) # in paired conversation, for example, would be 'chosen' and 'rejected'

        # since this will be pickled to avoid _LazyModule error in Hasher force
        # logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        data_args = raw_datasets.get_data_args()

        if dataset_type == "paired_conversation":
            if data_args.conversation_template:
                if data_args.conversation_template in PRESET_TEMPLATES.keys():
                    conversation_template = PRESET_TEMPLATES[data_args.conversation_template]
                else:
                    raise NotImplementedError(
                        f"Conversation template {data_args.conversation_template} is not supported yet."
                    )
            else:
                logger.warning("No conversation template provided. Using default template.")
                conversation_template = PRESET_TEMPLATES['empty']
                        
            logger.warning(f"Conversation template: {conversation_template}")
        else:
            raise NotImplementedError(
                f"Dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types for HFTextRegressionModel:\n"
                f"    {PAIRED_CONVERSATION_DATASET_DESCRIPTION}\n"
            )

        # Whether to truncate long sequences to fit into max_length
        use_truncation = False
        if model_args.use_lora or data_args.disable_group_texts:
            use_truncation = True
            
        tokenize_fn = tokenize_function
        tokenize_fn_kwargs = {
            "data_args": data_args,
            "tokenizer": self.tokenizer,
            "column_names": column_names,
            "conversation_template": conversation_template
        }
                           
        tokenize_kwargs = {}
        if not data_args.streaming:
            fingerprint = hashlib.md5(
                (
                    raw_datasets.get_fingerprint()
                    + str(self.tokenizer)
                    + ('###conversation_template=' + str(conversation_template) if "conversation" in dataset_type else "")
                    + f'###disable_group_texts={data_args.disable_group_texts}'
                    + f'###block_size={data_args.block_size}'
                ).encode("utf-8")
            ).hexdigest()
            tokenize_kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                "load_from_cache_file": not data_args.overwrite_cache,
                "desc": "Running tokenizer on dataset",
                "new_fingerprint": fingerprint,
            }

        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=column_names,
            fn_kwargs=tokenize_fn_kwargs,
            **tokenize_kwargs
        )
        return tokenized_datasets
            
            
    def save(self, dir, *args, **kwargs):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        dir :
            The directory to save model and tokenizer
        
        kwargs : Optional.
            Keyword arguments.    
        """
        self.get_tokenizer().save_pretrained(dir)
        self.get_backend_model().save_pretrained(dir)
