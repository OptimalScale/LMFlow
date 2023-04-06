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

import deepspeed
from filelock import FileLock
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_int8_training,
)
import nltk  # Here to have a nice missing dependency error message early on
import torch
import transformers
from transformers.deepspeed import HfDeepSpeedConfig

from transformers.testing_utils import CaptureLogger

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from lmflow.datasets.dataset import Dataset
from lmflow.models.encoder_decoder_model import EncoderDecoderModel
from lmflow.models.interfaces.tunable import Tunable

logger = logging.getLogger(__name__)

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
            try:
                nltk.data.find("tokenizers/punkt")
            except (LookupError, OSError):
                if is_offline_mode():
                    raise LookupError(
                        "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
                    )
                with FileLock(".lock") as lock:
                    nltk.download("punkt", quiet=True)

            config_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            if model_args.config_name:
                config = AutoConfig.from_pretrained(model_args.config_name, trust_remote_code=True, **config_kwargs)
            elif model_args.model_name_or_path:
                config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **config_kwargs)
            else:
                config = CONFIG_MAPPING[model_args.model_type]()
                logger.warning("You are instantiating a new config instance from scratch.")
                if model_args.config_overrides is not None:
                    logger.info(f"Overriding config: {model_args.config_overrides}")
                    config.update_from_string(model_args.config_overrides)
                    logger.info(f"New config: {config}")

            tokenizer_kwargs = {
                "cache_dir": model_args.cache_dir,
                "use_fast": model_args.use_fast_tokenizer,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            if model_args.tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True, **tokenizer_kwargs)
            elif model_args.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **tokenizer_kwargs)
            else:
                raise ValueError(
                    "You are instantiating a new tokenizer from scratch. This is"
                    " not supported by this script. You can do it from another"
                    " script, save it, and load it from here, using"
                    " --tokenizer_name."
                )

            if model_args.model_name_or_path:
                torch_dtype = (
                    model_args.torch_dtype
                    if model_args.torch_dtype in ["auto", None]
                    else getattr(torch, model_args.torch_dtype)
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForSeq2SeqLM.from_config(config)
                n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
                logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

            if model_args.use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=model_args.lora_r,
                    target_modules=["q_proj","v_proj"],
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()


            # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
            # on a small vocab and want a smaller embedding size, remove this test.
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
                if isinstance(tokenizer, MBartTokenizer):
                    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args["lang"]]
                else:
                    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args["lang"])

            if model.config.decoder_start_token_id is None:
                raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

            max_source_length = data_args["max_source_length"]
            if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < max_source_length
            ):
                if model_args.resize_position_embeddings is None:
                    logger.warning(
                        "Increasing the model's number of position embedding vectors from"
                        f" {model.config.max_position_embeddings} to {max_source_length}."
                    )
                    model.resize_position_embeddings(max_source_length)
                elif model_args.resize_position_embeddings:
                    model.resize_position_embeddings(max_source_length)
                else:
                    raise ValueError(
                        f"`--max_source_length` is set to {max_source_length}, but the model only has"
                        f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                        f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                        " model's position encodings by passing `--resize_position_embeddings`."
                    )
            prefix = data_args["source_prefix"] if data_args["source_prefix"] is not None else ""

            # # Preprocessing the datasets.
            # # We need to tokenize inputs and targets.
            # if training_args.do_train:
            #     if "train" not in raw_datasets:
            #         raise ValueError("--do_train requires a train dataset")
            #     column_names = raw_datasets["train"].column_names
            # elif training_args.do_eval:
            #     if "validation" not in raw_datasets:
            #         raise ValueError("--do_eval requires a validation dataset")
            #     column_names = raw_datasets["validation"].column_names
            # elif training_args.do_predict:
            #     if "test" not in raw_datasets:
            #         raise ValueError("--do_predict requires a test dataset")
            #     column_names = raw_datasets["test"].column_names
            # else:
            #     logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
            #     return

            self.config = config
            self.backend_model = model
            self.tokenizer = tokenizer
            self.tune_strategy = tune_strategy

        elif tune_strategy == 'none':
            dschf = HfDeepSpeedConfig(ds_config)
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
        
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force
        # logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
        if model_args.use_lora:
            self.tokenizer.pad_token = 1

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                if not model_args.use_lora:
                    output = self.tokenizer(examples[text_column_name])
                else:
                    output = self.tokenizer(
                        examples[text_column_name],
                        truncation=True,
                    )
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        data_args = raw_datasets.get_data_args()
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
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
