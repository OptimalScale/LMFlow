#!/usr/bin/env python
# coding=utf-8
import logging
import os,sys
# os.environ['CUDA_VISIBLE_DEVICES'] = "6"
from transformers.trainer_callback import TrainerControl, TrainerState
import wandb
from colorama import Fore,init
from typing import Optional, List

import torch
from datasets import load_dataset
from dataclasses import dataclass, field
from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback

from lmflow.utils.versioning import is_trl_available

if is_trl_available():
    from trl import (
        ModelConfig,
        SFTTrainer,
        DataCollatorForCompletionOnlyLM,
        SFTConfig,
        get_peft_config,
        get_quantization_config,
        get_kbit_device_map,
    )
    from trl.commands.cli_utils import TrlParser
else:
    raise ImportError("Please install trl package to use sft_summarizer.py")

@dataclass
class UserArguments:
    wandb_key: Optional[str] = field(
        default=None, metadata={"help": "User's own wandb key if there are multiple wandb accounts in your server"}
    )
    wandb_projectname: Optional[str] = field(
        default="huggingface_sft_summarizer", metadata={"help": "The name of project saved in wandb"}
    )

if __name__ == "__main__":
    # Initialize logging, tqdm and init
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    tqdm.pandas()
    init(autoreset=True)

    parser = TrlParser((UserArguments, SFTConfig, ModelConfig))
    user_args, sft_config, model_config = parser.parse_args_and_config()
    
    # Initialize wandb
    if user_args.wandb_key:
        wandb.login(key=user_args.wandb_key) # replace your own wandb key if there are multiple wandb accounts in your server
    else:
        wandb.init(mode="offline")
    wandb.init(project=user_args.wandb_projectname)

    # https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
    logging.debug(sft_config)
    logging.debug('-' * 50)
    logging.debug(model_config)
    logging.debug('-' * 50)
    logging.debug('cuda===> %s', os.environ['CUDA_VISIBLE_DEVICES'])


    if model_config.use_peft:
        use_peft = 'peft'
    else:
        use_peft = 'nopeft'


    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    logging.debug("torch_dtype===> %s", torch_dtype)
    if model_config.use_peft:
        quantization_config = None 
    else:
        quantization_config = get_quantization_config(model_config)
    logging.debug("quantization_config===> %s", quantization_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        local_files_only=True
    )
    logging.debug("model_kwargs: %s", model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################

    train_dataset = load_dataset("LukaMagic077/downsampled_below10k_arxiv_dataset_on_hub", split='train')
    val_dataset = load_dataset("LukaMagic077/downsampled_below10k_arxiv_dataset_on_hub", split='validation')
    # test_dataset = load_dataset("LukaMagic077/downsampled_below10k_arxiv_dataset_on_hub", split='test')
    
    # Get the size of training dataset
    train_dataset_size = len(train_dataset)
    # Get the size of validation dataset
    val_dataset_size = len(val_dataset)

    # Print the size of dataset
    logging.debug(f"Training dataset size: {train_dataset_size}")
    logging.debug(f"Validation dataset size: {val_dataset_size}")

    ################
    # Training
    ################
    
    # Define datacollector
    data_collector = DataCollatorForCompletionOnlyLM(
        instruction_template="article",
        response_template="abstract",
        tokenizer=tokenizer,
        mlm=False
    )

    class WandbCallback(TrainerCallback):
        def __init__(self, trainer):
            # trainer.model.to("cuda:0")
            self.model, self.tokenizer = trainer.model, trainer.tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.debug(Fore.GREEN + "entering callback=====>")
            logging.debug(self.tokenizer)
        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            logging.debug("current step %s", state.global_step)
            return super().on_save(args, state, control, **kwargs)


    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=sft_config,
        train_dataset= train_dataset,
        dataset_text_field="article",
        eval_dataset= val_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
