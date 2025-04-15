#!/usr/bin/env python
# coding=utf-8
"""Trainer for Peft models
"""

from __future__ import absolute_import
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments
import os
import numpy as np

class PeftTrainer(Trainer):
    def _save_checkpoint(self, _, trial, metrics=None):
        """ Don't save base model, optimizer etc.
            but create checkpoint folder (needed for saving adapter) """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value

                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

class PeftSavingCallback(TrainerCallback):
    """ Correctly save PEFT model and not full model """
    def _save(self, model, folder):
        if folder is None:
            folder = ""
        peft_model_path = os.path.join(folder, "adapter_model")
        model.save_pretrained(peft_model_path)

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
            control: TrainerControl, **kwargs):
        """ Save final best model adapter """
        self._save(kwargs['model'], state.best_model_checkpoint)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
            control: TrainerControl, **kwargs):
        """ Save intermediate model adapters in case of interrupted training """
        folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self._save(kwargs['model'], folder)
        
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       
        self._save(kwargs['model'], checkpoint_folder)

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control