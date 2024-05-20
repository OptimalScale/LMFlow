#!/usr/bin/env python
# coding=utf-8
"""This is a class called HFDecoderModel which is a wrapper around transformers model and
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

import hashlib
import logging
from typing import List, Union
import os, shutil
import deepspeed

from pathlib import Path

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_kbit_training
)

import torch
import transformers
from transformers.deepspeed import HfDeepSpeedConfig

from transformers.testing_utils import CaptureLogger

from transformers import BitsAndBytesConfig
import bitsandbytes

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.decoder_model import DecoderModel
from lmflow.models.interfaces.tunable import Tunable
from lmflow.utils.constants import (
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
    CONVERSATION_DATASET_DESCRIPTION
)
from lmflow.utils.conversation_template import ConversationTemplate, PRESET_TEMPLATES
from lmflow.utils.constants import CONVERSATION_ROLE_NAMES


logger = logging.getLogger(__name__)

MODELS_SUPPORT_FLASH_ATTENTION = [
    "LlamaForCausalLM",
    "GPTNeoForCausalLM",
    "GPT2ForCausalLM",
    "BloomForCausalLM"
]

GPU_SUPPORT_FLASH_ATTENTION = {
    "A100": ["LlamaForCausalLM", "GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"],
    "A40": ["GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"],
    "A6000": ["LlamaForCausalLM", "GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"]
}

try:
    import flash_attn
    if int(flash_attn.__version__.split(".")[0]) == 2:
        GPU_SUPPORT_FLASH_ATTENTION = {
            "A100": ["LlamaForCausalLM", "GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"],
            "A40": ["LlamaForCausalLM","GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"],
            "A6000": ["LlamaForCausalLM", "GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"]
        }
except Exception as e:
    if e.__class__ == ModuleNotFoundError:
        logger.warning(
            "flash_attn is not installed. Install flash_attn for better performance."
        )
    else:
        logger.warning(f'An error occurred when importing flash_attn, flash attention is disabled: {e}')

class HFDecoderModel(DecoderModel, Tunable):
    r"""
    Initializes a HFDecoderModel instance.

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
        Initializes a HFDecoderModel instance.
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
            logger.warning("The tokenizer_config.json file doesn't set the special tokens. Using default values: <unk>, <s>, </s> for unknown token, bos token and eos token respectively.")
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

        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
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
                    model = AutoModelForCausalLM.from_pretrained(
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
                    model = AutoModelForCausalLM.from_pretrained(
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
                model = AutoModelForCausalLM.from_config(config)
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
                self.backend_model = AutoModelForCausalLM.from_pretrained(
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
                        self.backend_model = AutoModelForCausalLM.from_pretrained(
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
                        self.backend_model = AutoModelForCausalLM.from_pretrained(
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
                    self.backend_model = AutoModelForCausalLM.from_pretrained(
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


    def tokenize(self, dataset, add_special_tokens=True, *args, **kwargs):
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
        column_names = list(hf_raw_datasets.features)

        # since this will be pickled to avoid _LazyModule error in Hasher force
        # logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        data_args = raw_datasets.get_data_args()

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
        tokenized_column_order = None       # Handles 1) and 2)
        label_columns = None                # Handles 3)
        if dataset_type == "text_only":
            tokenized_column_order = ["text"]
            label_columns = ["text"]
        elif dataset_type == "text2text":
            tokenized_column_order = ["input", "output"]
            label_columns = ["output"]
            add_special_tokens = False
        elif dataset_type == "conversation":
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
                f"dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types:\n"
                f"    1) {TEXT_ONLY_DATASET_DESCRIPTION}\n"
                f"    2) {TEXT2TEXT_DATASET_DESCRIPTION}\n"
                f"    3) {CONVERSATION_DATASET_DESCRIPTION}\n"
            )

        # Whether to truncate long sequences to fit into max_length
        use_truncation = False
        if model_args.use_lora or data_args.disable_group_texts:
            use_truncation = True
                           
        def tokenize_function(examples):
            num_example = len(examples[column_names[0]])
            token_dict = {
                "input_ids": [[] for _ in range(num_example)],
                "attention_mask": [[] for _ in range(num_example)],
                "labels": [[] for _ in range(num_example)],
            }
            with CaptureLogger(tok_logger) as cl:
                if dataset_type == "conversation":
                    for i in range(len(examples["messages"])):
                        messages = examples["messages"][i]
                        system = examples.get("system", [None] * num_example)[i]
                        tools = examples.get("tools", [None] * num_example)[i]
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

                        input_ids, labels = [], []
                        for turn_idx, (user_input, assistant_result) in enumerate(encoded_conversation):
                            input_ids += user_input + assistant_result
                            
                            if data_args.train_on_prompt:
                                labels += user_input + assistant_result
                            else:
                                labels += [-100] * len(user_input) + assistant_result
                            
                        token_dict["input_ids"][i].extend(input_ids)
                        token_dict["attention_mask"][i].extend([1] * len(input_ids))
                        token_dict["labels"][i].extend(labels)

                else:
                    for column_name in tokenized_column_order:
                        encoding = self.tokenizer(
                            examples[column_name],
                            add_special_tokens=add_special_tokens,
                            truncation=use_truncation,
                        )

                        if column_name in label_columns:
                            labels = encoding["input_ids"].copy()
                        else:
                            labels = [
                                [-100] * len(encoding["input_ids"][i])
                                for i in range(num_example)
                            ]

                        for i in range(num_example):
                            token_dict["input_ids"][i].extend(
                                encoding["input_ids"][i]
                            )
                            token_dict["attention_mask"][i].extend(
                                encoding["attention_mask"][i]
                            )
                            token_dict["labels"][i].extend(labels[i])

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
                        for key in ["input_ids", "attention_mask", "labels"]:
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
                        token_dict["labels"][i].extend(
                            [-100 for _ in range(pad_length)]
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


    def encode(self, input: Union[str, List[str]], *args, **kwargs ) -> Union[List[int], List[List[int]]]:
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
            if string input,return the tokenized inputs.
            "Hello,world!"-> [101, 7592, 1010, 2088, 102]
            if batch input,return {input_ids,attention_mask,token_type_ids}
            ["Hello,world!","Hello!"]-> {'input_ids': tensor([[  101,  7592,  1010,  2088,   102],...),'attention_mask': tensor([[1, 1, 1, 1, 1],[0,0,1,1,1]])}
        """
        if isinstance(input, list):
            return self.tokenizer(text=input, *args, **kwargs)#batch encode,will automatically do left padding
        elif isinstance(input, str):
            return self.tokenizer.encode(text=input, *args, **kwargs)
        else:
            raise NotImplementedError(f'type "{type(input)}" cannot be encoded')


    def decode(self, input, *args, **kwargs ) -> Union[str, List[str]]:
        """
        Perform decoding process of the tokenizer.
    
        Parameters
        ------------
        inputs : list or tensor.
            The token sequence.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The text decoded from the token inputs.
            if batch input,return the list of text
            [[101, 7592, 1010, 2088, 102],[101, 7592, 1010, 2088, 102]]-> ["Hello,world!","Hello,world!"
            if single input,return the text
            [101, 7592, 1010, 2088, 102]-> "Hello,world!"
        """
        if isinstance(input, List):
            input=torch.tensor(input)
        if input.dim()==2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)#batch_decode
        else:
            # Can be list of ints or a Tensor
            return self.tokenizer.decode(input, *args, **kwargs)


    def inference(self, inputs, use_accelerator=False, *args, **kwargs):
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
            if use_accelerator:
                outputs = self.backend_model.generate(
                    input_ids=inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    *args,
                    **kwargs
                )
            else:
                if self.device == "gpu":
                    outputs = self.ds_engine.module.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        *args,
                        **kwargs
                    )
                elif self.device == "cpu":
                    outputs = self.backend_model.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        *args,
                        **kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"device \"{self.device}\" is not supported"
                    )
        return outputs


    def merge_lora_weights(self):
        if self.model_args.use_lora and not self.model_args.use_qlora:
            self.get_backend_model().merge_and_unload()
        elif self.model_args.use_qlora:
            logger.warning("Reloading base model in 16-bit precision to merge adapter weights. NOTE: Your device must have"
                           "sufficient memory to reload the model in half-precision without quantization.")
            self.get_peft_without_qlora()
            self.get_backend_model().merge_and_unload()
        else:
            logger.warning("LoRA training is NOT enabled. Merging LoRA weights is not applicable.")

    def get_peft_without_qlora(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)


            self.get_backend_model().save_pretrained(tmpdirname)

            torch_dtype = (
                self.model_args.torch_dtype
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            config_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "revision": self.model_args.model_revision,
                "use_auth_token": True if self.model_args.use_auth_token else None,
            }
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, **config_kwargs)
            device_map = "auto"
            if os.environ.get('LOCAL_RANK') is not None:
                local_rank = int(os.environ.get('LOCAL_RANK','0'))
                device_map = {'': local_rank}

            self.backend_model_full = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code = self.model_args.trust_remote_code,
                attn_implementation="flash_attention_2" if self.model_args.use_flash_attention else None,
            )
        
            self.backend_model = PeftModel.from_pretrained(self.backend_model_full, tmpdirname)

    def save(self, dir, save_full_model=False, *args, **kwargs):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        dir :
            The directory to save model and tokenizer
            
        save_full_model : Optional.
            Whether to save full model.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """
        self.get_tokenizer().save_pretrained(dir)
        if save_full_model and self.model_args.use_lora:
            save_dtype = (
                torch.float16
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            self.backend_model_full.to(dtype=save_dtype).save_pretrained(dir)
            logger.warning(f"Save full model with dtype: {save_dtype}")
        else:
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
