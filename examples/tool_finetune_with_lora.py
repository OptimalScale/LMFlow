#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.

from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, HfArgumentParser, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import (
    LoraConfig,
    TaskType, 
    get_peft_model
)
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.utils.tool_bench.tool_conversation import SeparatorStyle
from lmflow.utils.tool_bench.model_adapter import get_conversation_template
from lmflow.utils.tool_bench.llama_condense_monkey_patch import replace_llama_with_condense
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.utils.peft_trainer import PeftTrainer, PeftSavingCallback

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
torch.set_printoptions(profile="full")


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def preprocess(
    sources,
    tokenizer: PreTrainedTokenizer,
    template: str="tool-llama"
) -> Dict:
    conv = get_conversation_template(template)
    if template == "tool-llama":
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
        if len(conv.roles) < 4:
            raise ValueError(f"Expected at least 4 roles in conversation template, got {len(conv.roles)}: {conv.roles}")
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Truncate conversations if they exceed 10000 characters
    # max_length = 1000
    # truncated_conversations = []
    # for conversation in conversations:
    #     if len(conversation) > max_length:
    #         print(f"Debug: Truncating conversation from length {len(conversation)} to {max_length}")
    #         truncated_conversations.append(conversation[:max_length])
    #     else:
    #         truncated_conversations.append(conversation)
    truncated_conversations = conversations

    # Tokenize conversations
    try:
        input_ids = tokenizer(
            truncated_conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    except Exception as e:
        print(f"Error during tokenization: {e}")
        for conv in truncated_conversations:
            print(f"Conversation length: {len(conv)}")
        raise e
    targets = input_ids.clone()
    
    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[-1] + ": "
    for conversation, target in zip(truncated_conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                continue
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            
            # only train on the last assistant reply, treat the history chat as instruction
            prefix = parts[:-1]
            instruction = ""
            for part in prefix:
                instruction += part
                instruction += sep

            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(instruction).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer, template="tool-llama"):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        self.template = template
        data_dict = preprocess(sources, tokenizer, self.template)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer, template="tool-llama"):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.template = template

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.template)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    print("Loading data...")
    raw_data = json.load(open(data_args.dataset_path, "r"))
    # if data_args.eval_data_path is not None:
    #     train_raw_data = raw_data
    #     eval_raw_data = json.load(open(data_args.eval_dataset_path, "r"))
    # else:
    # Split train/test
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.tool_conv_template)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.tool_conv_template)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def main():
    # Parse arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser(
        (ModelArguments, DatasetArguments, PipelineArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Argument of whether Use lora:", model_args.use_lora)
    
    # Initialization
    model = AutoModel.get_model(model_args)
    print("Type of model 1:", type(model))
    tokenizer = model.get_tokenizer()
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_peft_model(model.get_backend_model(), lora_config)
    print("Type of model 1:", type(model))
    model.print_trainable_parameters()
    if tokenizer.model_max_length < data_args.block_size:
        condense_ratio = int(data_args.block_size/tokenizer.model_max_length)
        # ratio = N means the sequence length is expanded by N, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
        replace_llama_with_condense(ratio=condense_ratio)

    # Prepare dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Finetuning
    # trainer = PeftTrainer(
    #     model=model.get_backend_model(), tokenizer=tokenizer, args=training_args, **data_module
    # )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # Save model
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
