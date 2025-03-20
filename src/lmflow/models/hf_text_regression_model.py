#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import os
import copy
import hashlib
import logging
from pathlib import Path
from typing import List, Union, Dict, Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from lmflow.args import ModelArguments
from lmflow.datasets.dataset import Dataset, KEY_SCORE
from lmflow.models.interfaces.tunable import Tunable
from lmflow.models.hf_model_mixin import HFModelMixin
from lmflow.models.text_regression_model import TextRegressionModel
from lmflow.tokenization.hf_text_regression_model import (
    paired_conversation_tokenize_function, 
    conversation_tokenize_function,
    tokenize_function,
    text_to_textlist_tokenize_function,
)
from lmflow.utils.conversation_template import PRESET_TEMPLATES
from lmflow.utils.constants import (
    PAIRED_CONVERSATION_DATASET_DESCRIPTION, 
    TEXT2TEXT_DATASET_DESCRIPTION,
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT_TO_TEXTLIST_DATASET_DESCRIPTION,
    CONVERSATION_DATASET_DESCRIPTION, 
)
from lmflow.utils.data_utils import RewardModelInferenceResultWithInput
from lmflow.utils.versioning import is_ray_available, is_vllm_available

if is_ray_available():
    import ray
    import ray.data
    
if is_vllm_available():
    from vllm import SamplingParams


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
        model_args: ModelArguments,
        tune_strategy: str='normal',
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
        assert model_args.arch_type == "text_regression", (
            f"Invalid model architecture type: {model_args.arch_type}. "
            f"Expected: text_regression"
        )
        config_additional_args = {"num_labels": 1}
        HFModelMixin.__init__(
            self,
            model_args=model_args,
            do_train=True if tune_strategy == "normal" else False,
            ds_config=ds_config,
            device=device,
            use_accelerator=use_accelerator,
            hf_auto_model_additional_args=config_additional_args,
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
        data_args = raw_datasets.get_data_args()
        
        # Whether to truncate long sequences to fit into max_length
        use_truncation = False
        if model_args.use_lora or data_args.disable_group_texts:
            use_truncation = True

        # Requires three types of information for tokenizing different datasets
        #   1) Which fields require tokenization, e.g.
        #        "text2float": "text", but not "float"
        #        "text2text": both "input" and "output"
        #   2) How will there tokenized sequence concatenated together, e.g.
        #        "text_only": "text" -> "text"
        #        "text2text": "input", "output" -> "input" + "output"
        #   3) Which fields require loss in final computation, e.g.
        #        "text_only": "text"
        #        "text2text": "output" only
        tokenize_fn = None
        tokenize_fn_kwargs = {
            "data_args": data_args,
            "tokenizer": self.tokenizer,
            "column_names": column_names,
        }            
        if dataset_type == "text_only":
            tokenize_fn = tokenize_function
            text_only_tokenize_fn_kwargs = {
                "tokenized_column_order": ["text"],
                "label_columns": ["text"],
                "add_special_tokens": add_special_tokens,
                "use_truncation": use_truncation,
            }
            tokenize_fn_kwargs.update(text_only_tokenize_fn_kwargs)
            
        elif dataset_type == "text2text":
            tokenize_fn = tokenize_function
            text2text_tokenize_fn_kwargs = {
                "tokenized_column_order": ["input", "output"],
                "label_columns": ["output"],
                "add_special_tokens": False,
                "use_truncation": use_truncation,
            }
            tokenize_fn_kwargs.update(text2text_tokenize_fn_kwargs)
            
        elif dataset_type in ["conversation", "paired_conversation"]:
            if dataset_type == "conversation":
                tokenize_fn = conversation_tokenize_function
            elif dataset_type == "paired_conversation":
                tokenize_fn = paired_conversation_tokenize_function
            
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
            tokenize_fn_kwargs["conversation_template"] = conversation_template
            logger.warning(f"Conversation template: {conversation_template}")
            
        elif dataset_type == "text_to_textlist":
            tokenize_fn = text_to_textlist_tokenize_function
            text_to_textlist_tokenize_fn_kwargs = {
                "add_special_tokens": False,
                "use_truncation": use_truncation,
            }
            tokenize_fn_kwargs.update(text_to_textlist_tokenize_fn_kwargs)
            
        else:
            raise NotImplementedError(
                f"Dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types for HFTextRegressionModel:\n"
                f"    1) [Inference]{TEXT_ONLY_DATASET_DESCRIPTION}\n"
                f"    2) [Inference]{TEXT2TEXT_DATASET_DESCRIPTION}\n"
                f"    3) [Training]{PAIRED_CONVERSATION_DATASET_DESCRIPTION}\n"
                f"    4) [Inference]{CONVERSATION_DATASET_DESCRIPTION}\n"
                f"    5) [Inference]{TEXT_TO_TEXTLIST_DATASET_DESCRIPTION}\n"
            )
  
        tokenize_kwargs = {}
        if not data_args.streaming:
            fingerprint = hashlib.md5(
                (
                    raw_datasets.get_fingerprint()
                    + str(self.tokenizer)
                    + f'###padding_side={self.tokenizer.padding_side}'
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


    def inference(
        self, 
        inputs, 
        release_gpu: bool = False,
        use_vllm: bool = False,
        **kwargs
    ) -> Union[List[float], SequenceClassifierOutputWithPast]:
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        inputs :
            The sequence used as a prompt for the generation or as model inputs to the model.
            When using vllm inference, this should be a string or a list of strings.
            When using normal inference, this should be a tensor.
        release_gpu : bool, optional
            Whether to release the GPU resource after inference, by default False.
        use_vllm : bool, optional
            Whether to use VLLM for inference, by default False.
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """
        if use_vllm:
            logger.warning(
                "VLLM inference is not supported for text regression model, using normal inference instead."
            )
            use_vllm = False
            
        if not self._activated:
            self.activate_model_for_inference(
                use_vllm=use_vllm,
                **kwargs,
            )
            
        if use_vllm:
            res = self.__vllm_inference(inputs, **kwargs)
        else:
            res = self.__inference(inputs, **kwargs)
            
        if release_gpu:
            self.deactivate_model_for_inference(use_vllm=use_vllm)
            
        return res


    def __inference(
        self, 
        inputs, 
        **kwargs
    ):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        inputs :
            The **tokenized** sequence used as a prompt for the generation or as model inputs to the model.
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """       
        with torch.no_grad():
            if self.use_accelerator:
                outputs = self.backend_model(
                    input_ids=inputs,
                    **kwargs,
                )
            else:
                if self.device == "gpu":
                    outputs = self.ds_engine.module(
                        input_ids=inputs,
                        synced_gpus=True,
                        **kwargs,
                    )
                elif self.device == "cpu":
                    outputs = self.backend_model(
                        input_ids=inputs,
                        synced_gpus=True,
                        **kwargs,
                    )
                else:
                    raise NotImplementedError(
                        f"device \"{self.device}\" is not supported"
                    )
        
        if kwargs.get('return_input', False):
            outputs = {"input": inputs, "output": outputs}
        
        return outputs
    
    
    def __vllm_inference(
        self, 
        inputs: Union[str, List[str]],
        sampling_params: Optional['SamplingParams'] = None,
        **kwargs,
    ) -> Union[List[List[str]], List[List[List[int]]]]:
        """Perform VLLM inference process of the model.

        Parameters
        ----------
        inputs : Union[str, List[str]]
            Prompt(s), string or a list of strings.
        sampling_params : Optional[SamplingParams], optional
            vllm SamplingParams object, by default None.

        Returns
        -------
        """
        raise NotImplementedError(
            "VLLM inference is not supported for text regression model."
        )
        
        
    def prepare_inputs_for_inference(
        self,
        dataset: Dataset,
        enable_distributed_inference: bool = False,
        use_vllm: bool = False,
        **kwargs,
    ) -> Union[Dataset, 'ray.data.Dataset']:
        if use_vllm:
            raise NotImplementedError(
                "VLLM inference is not supported for text regression model."
            )
            
        inference_inputs = self.tokenize(dataset)
                
        if enable_distributed_inference:
            if not is_ray_available():
                raise ValueError(
                    'Ray is not available. Please install ray via `pip install -e ".[ray]"`.'
                )
            
            inference_inputs.sanity_check(drop_invalid=True)
            inference_inputs = inference_inputs.get_backend_dataset()
            inference_inputs = ray.data.from_items(inference_inputs) 
            # -> Dict[str, np.ndarray]
            # Example (batch size=2):
            # {'input': array(['...','...'], dtype=object),
            #  'output': array([array(["...", "..."], dtype=object), array(['...','...'], dtype=object)], dtype=object),
            #  'input_ids': array(
            #      [
            #          array([array([    27,     91,    882, ..., 128256, 128256, 128256]),
            #                 array([    27,     91,    882, ..., 128256, 128256, 128256])],
            #                dtype=object),
            #          array([array([    27,     91,    882, ..., 128256, 128256, 128256]),
            #                 array([    27,     91,    882, ..., 128256, 128256, 128256])],
            #                dtype=object)
            #      ], dtype=object)}
        
        return inference_inputs
            
    
    @staticmethod
    def postprocess_inference_outputs(
        dataset: Dataset,
        scores: Union[List[float], List[List[float]]],
    ):
        output_dict = {"type": "", "instances": []}
        if dataset.get_type() == "text_to_textlist":
            output_dict["type"] = "text_to_scored_textlist"
            for idx, instance in enumerate(dataset.get_backend_dataset()):
                if len(instance["output"]) < 2:
                    logger.warning(f"Instance {idx} has less than 2 outputs, skipping.")
                output_dict["instances"].append(
                    {
                        "input": instance["input"],
                        "output": [{"text": text} for text in instance["output"]],
                    }
                )
        else:
            raise NotImplementedError(f"Dataset type {dataset.get_type()} is not supported for reward model inference.")
        
        for i, instance_scores in enumerate(scores):
            for j, score in enumerate(instance_scores):
                output_dict["instances"][i]["output"][j][KEY_SCORE] = score
        
        output_dataset_args = copy.deepcopy(dataset.data_args)
        output_dataset_args.dataset_path = None
        output_dataset_args.dataset_name = f"{output_dataset_args.dataset_name}_scored"
        output_dataset = Dataset(output_dataset_args)
        output_dataset = output_dataset.from_dict(output_dict)

        return output_dataset
    
    
    @staticmethod
    def postprocess_distributed_inference_outputs(
        dataset: Dataset,
        inference_result: List[RewardModelInferenceResultWithInput],
    ):
        output_dict = {"type": "text_to_scored_textlist", "instances": inference_result}
        output_dataset_args = copy.deepcopy(dataset.data_args)
        output_dataset_args.dataset_path = None
        output_dataset_args.dataset_name = f"{output_dataset_args.dataset_name}_scored"
        output_dataset = Dataset(output_dataset_args)
        output_dataset = output_dataset.from_dict(output_dict)
        
        return output_dataset
        

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
