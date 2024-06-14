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
from lmflow.pipeline.utils.ppov2_trainer import PPOTrainer


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
        num_train_samples = len(train_dataset)
        logger.info(f"Number of train samples: {num_train_samples}")

        if self.finetuner_args.do_train and self.data_args.max_train_samples is not None:
            max_train_samples = min(num_train_samples, self.data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        if self.finetuner_args.do_eval:
            if self.finetuner_args.eval_dataset_path:
                eval_dataset_args = deepcopy(self.data_args)
                eval_dataset_args.dataset_path = self.finetuner_args.eval_dataset_path
                eval_dataset = Dataset(eval_dataset_args)
                with self.finetuner_args.main_process_first(desc="dataset map tokenization"):
                    tokenized_dataset = model.tokenize(eval_dataset)
                    if self.data_args.disable_group_texts:
                        lm_dataset = tokenized_dataset
                    else:
                        lm_dataset = self.group_text(
                            tokenized_dataset,
                            model_max_length=model.get_max_length(),
                        )
                eval_dataset = lm_dataset.get_backend_dataset()
            else:
                num_eval_sampels = int(num_train_samples * 0.2)
                eval_dataset = train_dataset.select(range(num_train_samples - num_eval_sampels, num_train_samples))
                train_dataset = train_dataset.select(range(num_train_samples - num_eval_sampels))
                logger.warning(f"You've set `do_eval=True` but haven't provided an `eval_dataset_path`. "
                               "Using 0.2 of the training dataset for evaluation (These samples "
                               "will not be used for training). If you want to use a different dataset "
                               "for evaluation, please provide the path to the dataset using")
            logger.info(f"Number of eval samples: {len(eval_dataset)}")
        
        # 2. prepare trainer
        trainer = PPOTrainer(
            config=self.finetuner_args,
            tokenizer=model.get_tokenizer(),
            policy=model.get_backend_model(),
            ref_policy=ref_model.get_backend_model(),
            reward_model=reward_model.get_backend_model(),
            value_model=value_model.get_backend_model(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # 3. training
        if self.finetuner_args.do_train:
            # TODO: checkpointing
            trainer.train()
            trainer.save_model(self.finetuner_args.output_dir)
            print("Model saved to %s", self.finetuner_args.output_dir)
            if self.finetuner_args.push_to_hub:
                print('push to hub')
                trainer.push_to_hub()
            trainer.generate_completions()

        return model