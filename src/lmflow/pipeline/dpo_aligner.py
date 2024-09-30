#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/4/2024 21:12
# @Author  : Yu Li
# @Site    :
# @File    : dpo_pipeline.py
import os
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from lmflow.pipeline.base_aligner import BaseAligner
from lmflow.utils.versioning import is_trl_available

if is_trl_available():
    from trl import DPOTrainer
else:
    raise ImportError("Please install trl package to use dpo_aligner.py")


def get_paired_dataset(
        data_root: str,
        data_dir: str,
        sanity_check: bool = False,
        cache_dir: Optional[str] = None,
        num_proc=24,
) -> Dataset:
    """Load dataset and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    data_path = Path(data_root) / data_dir
    data_files = [
        x.absolute().as_posix()
            for x in data_path.glob("*.json")
    ]
    dataset = load_dataset(
        path=data_root,
        split="train",
        data_files=data_files,
        cache_dir=cache_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


class DPOAligner(BaseAligner):
    def __init__(self, model_args, data_args, aligner_args):
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = aligner_args
        self.train_dataset = None
        self.eval_dataset = None

    def _initialize_trainer(self, model, tokenizer):
        peft_config = LoraConfig(
            r=self.model_args.lora_r,
            lora_alpha=self.model_args.lora_alpha,
            lora_dropout=self.model_args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        training_args = TrainingArguments(
            per_device_train_batch_size=self.aligner_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.aligner_args.per_device_eval_batch_size,
            max_steps=self.aligner_args.max_steps,
            logging_steps=self.aligner_args.logging_steps,
            save_steps=self.aligner_args.save_steps,
            gradient_accumulation_steps=self.aligner_args.gradient_accumulation_steps,
            gradient_checkpointing=self.aligner_args.gradient_checkpointing,
            learning_rate=self.aligner_args.learning_rate,
            evaluation_strategy="steps",
            eval_steps=self.aligner_args.eval_steps,
            output_dir=self.aligner_args.output_dir,
            report_to=self.aligner_args.report_to,
            lr_scheduler_type=self.aligner_args.lr_scheduler_type,
            warmup_steps=self.aligner_args.warmup_steps,
            optim=self.aligner_args.optimizer_type,
            bf16=True,
            remove_unused_columns=False,
            run_name=self.aligner_args.run_name,
            ddp_find_unused_parameters=False,
            # gradient_checkpointing_kwargs=dict(use_reentrant=self.aligner_args.gradient_checkpointing_use_reentrant),
            seed=self.aligner_args.seed,
        )
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=self.aligner_args.beta,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if self.eval_dataset else None,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_prompt_length=self.aligner_args.beta,
            max_length=self.aligner_args.max_length,
        )
        return dpo_trainer

    def _load_dataset(self):
        # load training set
        self.train_dataset = get_paired_dataset(data_root=self.data_args.dataset_path,
                                                data_dir="train",
                                                sanity_check=self.aligner_args.sanity_check)
        self.train_dataset = self.train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= self.aligner_args.max_length
                      and len(x["prompt"]) + len(x["rejected"]) <= self.aligner_args.max_length
        )
        # load evaluation set
        if self.aligner_args.eval_dataset_path:
            self.eval_dataset = get_paired_dataset(data_root=self.aligner_args.eval_dataset_path,
                                                data_dir="test",
                                                sanity_check=True)
            self.eval_dataset = self.eval_dataset.filter(
                lambda x: len(x["prompt"]) + len(x["chosen"]) <= self.aligner_args.max_length
                        and len(x["prompt"]) + len(x["rejected"]) <= self.aligner_args.max_length
            )

    def align(self, model, dataset, reward_model):
        tokenizer = model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self._load_dataset()

        wrapped_model = model
        model = model.get_backend_model()

        dpo_trainer = self._initialize_trainer(model, tokenizer)
        dpo_trainer.train()
        dpo_trainer.save_model(self.aligner_args.output_dir)

        # 7. save
        output_dir = os.path.join(self.aligner_args.output_dir, "final_checkpoint")
        dpo_trainer.model.save_pretrained(output_dir)




