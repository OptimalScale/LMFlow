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
from lmflow.models.text_regression_model import TextRegressionModel
from lmflow.utils.conversation_template import ConversationTemplate, PRESET_TEMPLATES
from lmflow.utils.constants import PAIRED_CONVERSATION_DATASET_DESCRIPTION, CONVERSATION_ROLE_NAMES


logger = logging.getLogger(__name__)


class HFTextRegressionModel(TextRegressionModel, Tunable):
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

        # See more about loading any type of standard or custom dataset (from
        # files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer
        #
        # Distributed training: The .from_pretrained methods guarantee that
        # only one local process can concurrently download model & vocab.

        self.device = device
        self.model_args = model_args
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "trust_remote_code": model_args.trust_remote_code,
        }
        
        try:
            if model_args.tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
            elif model_args.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
            else:
                raise ValueError(
                    "You are instantiating a new tokenizer from scratch. This is"
                    " not supported by this script. You can do it from another"
                    " script, save it, and load it from here, using"
                    " --tokenizer_name."
                )

        except RecursionError:
            logger.warning(
                "The tokenizer_config.json file doesn't set the special tokens. Using default values: "
                "<unk>, <s>, </s> for unknown token, bos token and eos token respectively.")
            if model_args.tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, unk_token="<unk>",
                                                    bos_token="<s>",
                                                    eos_token="</s>",
                                                    **tokenizer_kwargs)
            elif model_args.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, unk_token="<unk>",
                                                    bos_token="<s>",
                                                    eos_token="</s>",
                                                    **tokenizer_kwargs)
            else:
                raise ValueError(
                    "You are instantiating a new tokenizer from scratch. This is"
                    " not supported by this script. You can do it from another"
                    " script, save it, and load it from here, using"
                    " --tokenizer_name."
                )
            
        self.tokenizer = tokenizer 
        self.tokenizer.truncation_side = model_args.truncation_side
        self.tokenizer.model_max_length = model_args.model_max_length
        
        if model_args.torch_dtype in ["auto", None, "bf16", "bfloat16"]:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = getattr(torch, model_args.torch_dtype)
            logger.warning(
                f"InstructGPT uses torch.bfloat16 for reward model, but you"
                f" are using {torch_dtype} for your reward model init. Ignore"
                f" this warning if it is intended.")
        logger.debug(f"torch_dtype on init: {torch_dtype}")

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "trust_remote_code": model_args.trust_remote_code,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")

        #position interpolation
        if model_args.do_rope_scaling:
            if "LlamaForCausalLM" in config.architectures:
                from lmflow.utils.position_interpolation.llama_rope_scaled_monkey_patch import (
                        replace_llama_with_condense,
                )
                replace_llama_with_condense(model_args.rope_pi_ratio, model_args.rope_ntk_ratio)

        if tune_strategy == 'normal':
            if model_args.model_name_or_path:
                compute_dtype = torch_dtype
                device_map = "auto"
                if os.environ.get('LOCAL_RANK') is not None:
                    local_rank = int(os.environ.get('LOCAL_RANK','0'))
                    device_map = {'': local_rank}

                if model_args.use_qlora:
                    model_args.use_lora = True
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=model_args.bits == 4,
                        load_in_8bit=model_args.bits == 8,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=model_args.double_quant,
                        bnb_4bit_quant_type=model_args.quant_type,
                    )
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        quantization_config=quant_config if model_args.use_qlora else None,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        torch_dtype=torch_dtype,
                        trust_remote_code = model_args.trust_remote_code,
                        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
                    )
                #for deepspeed zero3, we don't need to specify device_map
                except:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        quantization_config=quant_config if model_args.use_qlora else None,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        torch_dtype=torch_dtype,
                        trust_remote_code = model_args.trust_remote_code,
                        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
                    )
                if model_args.use_qlora:
                    model.gradient_checkpointing_enable()
                    model = prepare_model_for_kbit_training(model)
            else:
                model = AutoModelForSequenceClassification.from_config(config)
                n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
                logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
            self.backend_model_full = model
            
            if model_args.use_lora:
                if model_args.lora_target_modules:
                    lora_target_modules = model_args.lora_target_modules
                else:
                    lora_target_modules = None
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=lora_target_modules,
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

            # We resize the embeddings only when necessary to avoid index errors.
            # If you are creating a model from scratch on a small vocab and want a
            # smaller embedding size, remove this test.
            with deepspeed.zero.GatheredParameters(model.get_input_embeddings().weight, modifier_rank=None):
                weights = model.get_input_embeddings().weight
                embedding_size = weights.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            self.config = config
            self.backend_model = model
            self.tune_strategy = tune_strategy

        elif tune_strategy == 'none':
            if use_accelerator:
                peft_model_id = model_args.lora_model_path
                self.backend_model = AutoModelForSequenceClassification.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        device_map="auto",
                        offload_folder="offload",
                        offload_state_dict=True,
                        torch_dtype=torch_dtype,
                        load_in_8bit = model_args.use_int8,
                        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
                    )
                if peft_model_id is not None:
                    self.backend_model = PeftModel.from_pretrained(
                        self.backend_model, 
                        peft_model_id,
                    )
                self.tokenizer.padding_side = "left"
            else:
                dschf = HfDeepSpeedConfig(ds_config)
                peft_model_id = model_args.lora_model_path
                # NOTE: Currently offload is not supported by llama
                if config.model_type == "llama" and model_args.use_ram_optimized_load:
                    logger.warning(
                        "llama does not support RAM optimized load. Automatically"
                        " use original load instead."
                    )
                    model_args.use_ram_optimized_load = False

                if model_args.use_ram_optimized_load and peft_model_id is None:
                    try:
                        # RAM-optimized load
                        self.backend_model = AutoModelForSequenceClassification.from_pretrained(
                            model_args.model_name_or_path,
                            config=config,
                            device_map="auto",
                            offload_folder="offload",
                            offload_state_dict=True,
                            torch_dtype=torch_dtype,
                            attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
                        )
                    except:
                        logger.warning(
                            "Failed to use RAM optimized load. Automatically"
                            " use original load instead."
                        )
                        # Normal load
                        self.backend_model = AutoModelForSequenceClassification.from_pretrained(
                            model_args.model_name_or_path,
                            config=config,
                            torch_dtype=torch_dtype,
                            attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
                        )
                else:
                    if peft_model_id is not None:
                        logger.warning(
                            "LoRA does not support RAM optimized load currently."
                            " Automatically use original load instead."
                        )
                    self.backend_model = AutoModelForSequenceClassification.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
                    )

                self.backend_model_full = self.backend_model
                if peft_model_id is not None:
                    self.backend_model = PeftModel.from_pretrained(
                        self.backend_model, peft_model_id
                    )
  
                self.tokenizer.padding_side = "left" #necessary for llama, gpt2 and other decoder models
                
                if device == "gpu":
                    deepspeed.init_distributed()
                    self.ds_engine = deepspeed.initialize(model=self.backend_model, config_params=ds_config)[0]
                    self.ds_engine.module.eval()

        elif tune_strategy == 'adapter':
            raise NotImplementedError('adapter tune strategy not implemented')

        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.backend_model.config.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
    
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
            conversation_template: ConversationTemplate = kwargs.get("conversation_template")
            if conversation_template:
                if data_args.conversation_template:
                    logger.warning("You specified conversation_template in both model.tokenize() and data_args. "
                                   "Template in model.tokenize() will be used.")
            else:
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
                           
        def tokenize_function(examples):
            num_example = len(examples[column_names[0]])
            token_dict = {}
            for column_name in column_names:
                token_dict[f"input_ids_{column_name}"] = [[] for _ in range(num_example)]
                token_dict[f"attention_mask_{column_name}"] = [[] for _ in range(num_example)]
                
            with CaptureLogger(tok_logger) as cl:
                if dataset_type == "paired_conversation":
                    for i in range(len(num_example)):
                        for column_name in column_names:
                            messages = examples[column_name][i]["messages"]
                            system = examples[column_name][i].get("system", [None] * num_example)
                            tools = examples[column_name][i].get("tools", [None] * num_example)
                            if len(messages) < 2 or messages[0]['role'] != CONVERSATION_ROLE_NAMES['user']:
                                tok_logger.warning(
                                    "Invalid instance encountered. Either the conversation has less than "
                                    "one round or the first message is not from the user."
                                )
                                continue
                        
                            if len(messages) % 2 != 0:
                                logger.warning(
                                    "The number of messages is not even, the last message will be ignored."
                                )
                                messages = messages[:-1]
                            
                            encoded_conversation = conversation_template.encode_conversation(
                                tokenizer=self.tokenizer,
                                messages=messages,
                                system=system,
                                tools=tools,
                            )

                            input_ids = []
                            for turn_idx, (user_input, assistant_result) in enumerate(encoded_conversation):
                                input_ids += user_input + assistant_result
                                
                            token_dict[f"input_ids_{column_name}"][i].extend(input_ids)
                            token_dict[f"attention_mask_{column_name}"][i].extend([1] * len(input_ids))

                else:
                    raise NotImplementedError(
                        f"Dataset type {dataset_type} is not supported yet."
                    )
                    

            if data_args.disable_group_texts:
                block_size_warning_num = 0
                for i in range(num_example):
                    block_size = data_args.block_size
                    max_length = min(block_size, self.get_max_length())
                    pad_length = max_length - len(token_dict["input_ids"][i])
                    if block_size < self.get_max_length():
                        block_size_warning_num += 1
                    if pad_length < 0:
                        # Truncates too long samples
                        for key in ["input_ids", "attention_mask"]:
                            token_dict[key][i] = token_dict[key][i][:pad_length]
                    else:
                        # Pads too short samples
                        pad_token_id = self.tokenizer.pad_token_id
                        token_dict["input_ids"][i].extend(
                            [pad_token_id for _ in range(pad_length)]
                        )
                        token_dict["attention_mask"][i].extend(
                            [0 for _ in range(pad_length)]
                        )
                if block_size_warning_num > 0:
                    logger.warning(
                        f"There are {block_size_warning_num} of {num_example} samples where"
                        f"block_size {block_size} < model_max_length"
                        f" {self.get_max_length()}, use block_size"
                        " for maximum tokenized sequence length"
                    )

            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return token_dict

        if not data_args.streaming:
            fingerprint = raw_datasets.get_fingerprint()
            new_fingerprint = hashlib.md5(
                (
                    fingerprint
                    + str(self.tokenizer)
                    + str(conversation_template) if "conversation" in dataset_type else ""
                    + f'###disable_group_texts={data_args.disable_group_texts}'
                    + f'###block_size={data_args.block_size}'
                ).encode("utf-8")
            ).hexdigest()

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                new_fingerprint=new_fingerprint,
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
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
