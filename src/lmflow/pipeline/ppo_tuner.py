#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import sys
import logging
from typing import Optional
from copy import deepcopy

from lmflow.datasets import Dataset
from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.utils.ppo_trainer import PPOTrainer


logger = logging.getLogger(__name__)


class PPOTuner(Finetuner):
    """Initializes the `PPOTuner` class.

    Parameters
    ----------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
        
    reward_model_args : RewardModelArguments object.
        Contains the arguments required to load the reward model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : RewardModelingArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.
    """
    def __init__(
        self, 
        model_args, 
        data_args, 
        finetuner_args, 
        *args, 
        **kwargs
    ):
        super().__init__(model_args, data_args, finetuner_args, *args, **kwargs)
        self.reward_model_args = kwargs.get('reward_model_args')
        
    
    def tune(
        self,
        model: HFDecoderModel,
        ref_model: HFDecoderModel,
        reward_model: HFTextRegressionModel,
        value_model: HFTextRegressionModel,
        dataset,
        transform_dataset_in_place=True,
        **kwargs
    ):
        # 0. basic init
        if not transform_dataset_in_place:
            dataset = deepcopy(dataset)
            
        # 1. prepare dataset
        with self.finetuner_args.main_process_first(desc="dataset map tokenization"):
            tokenized_dataset = model.tokenize(dataset)
            if self.data_args.disable_group_texts:
                lm_dataset: Dataset = tokenized_dataset
            else:
                lm_dataset: Dataset = self.group_text(
                    tokenized_dataset,
                    model_max_length=model.get_max_length(),
                )
        train_dataset = lm_dataset.get_backend_dataset()
        logger.info(f"Number of train samples: {len(train_dataset)}")

        if self.finetuner_args.do_train and self.data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        if self.finetuner_args.do_eval:
            logger.warning("Currently eval for RLHF is not supported.")
        
        # 2. prepare trainer
        trainer = PPOTrainer(
            config=self.finetuner_args,
            tokenizer=model.get_tokenizer(),
            policy=model.get_backend_model(),
            ref_policy=ref_model.get_backend_model(),
            reward_model=reward_model.get_backend_model(),
            value_model=value_model.get_backend_model(),
            train_dataset=train_dataset,
            eval_dataset=None
        )
        
        # 3. training
        if self.finetuner_args.do_train:
            # TODO: checkpointing
            trainer.train()
            trainer.save_model(self.finetuner_args.output_dir)
            trainer.push_to_hub()
            trainer.generate_completions()

        return model