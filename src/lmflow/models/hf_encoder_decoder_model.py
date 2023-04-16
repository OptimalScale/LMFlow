#!/usr/bin/env python
# coding=utf-8
"""This is a class called HFEncoderDecoder which is a wrapper around transformers model and
tokenizer classes. It has several methods such as __init__, tokenize, and train that are 
used for training and fine-tuning the model. The __init__ method takes in several arguments
such as model_args, tune_strategy, and ds_config, which are used to load the pretrained 
model and tokenizer, and initialize the training settings.

The tokenize method is used to tokenize the input text and return the input IDs and attention
masks that can be fed to the model for training or inference.

This class supports different tune_strategy options such as 'normal', 'none', 'lora', and
'adapter', which allow for different fine-tuning settings of the model. However, the 'lora'
and 'adapter' strategies are not yet implemented.

Overall, this class provides a convenient interface for loading and fine-tuning transformer
models and can be used for various NLP tasks such as language modeling, text classification,
and question answering.
"""
import logging
from typing import List, Union
import re

import deepspeed
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_int8_training,
)
import torch
import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers.testing_utils import CaptureLogger

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
)
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from lmflow.datasets.dataset import Dataset
from lmflow.models.encoder_decoder_model import EncoderDecoderModel
from lmflow.models.interfaces.tunable import Tunable

logger = logging.getLogger(__name__)

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
    
class HFEncoderDecoderModel(EncoderDecoderModel, Tunable):
    r"""
    Initializes a HFEncoderDecoderModel instance.

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
        *args,
        **kwargs
    ):
        """
        Initializes a HFEncoderDecoderModel instance.
        :param model_args: dictionary with model arguments such as model name, path, revision, etc.
        :param tune_strategy: tuning strategy: normal, none, lora or adapter
        :param ds_config: deepspeed configuration for distributed training
        """

        # See more about loading any type of standard or custom dataset (from
        # files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer
        #
        # Distributed training: The .from_pretrained methods guarantee that
        # only one local process can concurrently download model & vocab.

        data_args = kwargs
        self.data_args = data_args
        self.model_args = model_args

        if tune_strategy == 'normal':
            raise NotImplementedError(
                f"tune_strategy \"{tune_strategy}\" is not supported"
            )
        elif tune_strategy == 'none':
            dschf = HfDeepSpeedConfig(ds_config)
            if model_args.model_name_or_path == 'THUDM/chatglm-6b':
                self.backend_model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            else:
                self.backend_model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
            peft_model_id = model_args.lora_model_path
            if peft_model_id is not None:
                self.backend_model = PeftModel.from_pretrained(
                    self.backend_model, peft_model_id
                )

            deepspeed.init_distributed()
            self.ds_engine = deepspeed.initialize(model=self.backend_model, config_params=ds_config)[0]
            self.ds_engine.module.eval()

        elif tune_strategy == 'adapter':
            raise NotImplementedError('adapter tune strategy not implemented')


    def tokenize(self, dataset, *args, **kwargs):
        """
        Tokenize the full dataset.
    
        Parameters
        ------------
        dataset : 
            Text dataset.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset.
        """
        model_args = self.model_args
        data_args = self.data_args
        text_column = "input"
        summary_column = "output"
        prefix = data_args["source_prefix"] if data_args["source_prefix"] is not None else ""

        # A list of all multilingual tokenizer which require lang attribute.
        MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

        if isinstance(self.tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
            assert (
                data_args["lang"] is not None
            ), f"{self.tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

            self.tokenizer.src_lang = data_args["lang"]
            self.tokenizer.tgt_lang = data_args["lang"]

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
            forced_bos_token_id = (
                self.tokenizer.lang_code_to_id[data_args["forced_bos_token"]] if data_args["forced_bos_token"] is not None else None
            )
            self.model.config.forced_bos_token_id = forced_bos_token_id
        
        # Temporarily set max_target_length for training.
        max_target_length = data_args["max_target_length"]
        padding = "max_length" if data_args["pad_to_max_length"] else False

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        # TODO: DO WE NEED THIS?
        # since this will be pickled to avoid _LazyModule error in Hasher force
        # logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
        if model_args.use_lora:
            self.tokenizer.pad_token = 1

        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        def preprocess_function(examples):
            # remove pairs where at least one record is None

            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] and examples[summary_column][i]:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = self.tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

            # Tokenize targets with the `text_target` keyword argument
            labels = self.tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
    
        data_args = raw_datasets.get_data_args()

        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        return tokenized_datasets


    def encode(self, input: Union[str, List[str]], *args, **kwargs ) -> List[int]:
        """
        Perform encoding process of the tokenizer.
    
        Parameters
        ------------
        inputs : str or list.
            The text sequence.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The tokenized inputs.
        """
        return self.tokenizer.encode(text=input, *args, **kwargs)
    

    def decode(self, input, *args, **kwargs ) -> List[int]:
        """
        Perform decoding process of the tokenizer.
    
        Parameters
        ------------
        inputs : list.
            The token sequence.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The text decoded from the token inputs.
        """
        return self.tokenizer.decode(input, *args, **kwargs)
    
    def inference(self, inputs, *args, **kwargs):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        inputs :
            The sequence used as a prompt for the generation or as model inputs to the model.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """


        with torch.no_grad():
            if self.model_args.model_name_or_path == 'THUDM/chatglm-6b':
                raw_input = kwargs.pop('raw_input', None)
                history = []
                logits_processor = None

                if history is None:
                    history = []
                if logits_processor is None:
                    logits_processor = LogitsProcessorList()
                logits_processor.append(InvalidScoreLogitsProcessor())
                # gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                #             "temperature": temperature, "logits_processor": logits_processor, **kwargs}
                if not history:
                    prompt = raw_input
                else:
                    prompt = ""
                    for i, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), raw_input)
                inputs = self.tokenizer([prompt], return_tensors="pt")
                inputs = inputs.to("cuda")
                for outputs in self.backend_model.stream_generate(**inputs, **kwargs):
                    # outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
                    print(f"outputs = {outputs}")
                    # response = self.tokenizer.decode(outputs)
                    # response = self.process_response(response)
                    # new_history = history + [(raw_input, response)]
                    # yield response, new_history

                # for response, history in self.backend_model.stream_chat(self.tokenizer, raw_input, history=history):
                #     outputs = response
                    # Prints characters in the buffer
                    # new_print_index = print_index
                    # for char in response[print_index:]:
                    #     if end_string is not None and char == end_string[0]:
                    #         if new_print_index + len(end_string) >= len(response):
                    #             break

                    #     new_print_index += 1
                    #     print(char, end="", flush=True)

                    # print_index = new_print_index

                
                # logits_processor = None
                # if history is None:
                #     history = []
                # if logits_processor is None:
                #     logits_processor = LogitsProcessorList()

                # logits_processor.append(InvalidScoreLogitsProcessor())                
                # kwargs.update({"logits_processor": logits_processor})
                # inputs = self.tokenizer([raw_input], return_tensors="pt")
                # inputs = inputs.to("cuda")
                # outputs = self.ds_engine.module.generate(**inputs, **kwargs)

            else:
                outputs = self.ds_engine.module.generate(
                    input_ids=inputs,
                    synced_gpus=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    *args,
                    **kwargs
                )
        return outputs


    def get_max_length(self):
        """
        Return max acceptable input length in terms of tokens.
        """
        return self.tokenizer.model_max_length


    def get_tokenizer(self):
        """
        Return the tokenizer of the model.
        """
        return self.tokenizer


    def get_backend_model(self):
        """
        Return the backend model.
        """
        return self.backend_model