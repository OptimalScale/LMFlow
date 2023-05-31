#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import copy
import logging
import os
import sys

import datasets
import transformers
import evaluate
from itertools import chain
from transformers import (
    Trainer,
    default_data_collator,
    set_seed,
)
from copy import deepcopy
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_tuner import BaseTuner
from lmflow.pipeline.utils.peft_trainer import PeftTrainer, PeftSavingCallback


logger = logging.getLogger(__name__)


class Finetuner(BaseTuner):
    """
    Initializes the `Finetuner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : FinetunerArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, finetuner_args, *args, **kwargs):
        
        self.model_args = model_args
        self.data_args = data_args
        self.finetuner_args = finetuner_args

        # Sending telemetry. Tracking the example usage helps us better
        # allocate resources to maintain them. The information sent is the one
        # passed as arguments along with your Python/PyTorch versions.
        send_example_telemetry("run_clm", model_args, data_args)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = finetuner_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {finetuner_args.local_rank},"
            f" device: {finetuner_args.device},"
            f" n_gpu: {finetuner_args.n_gpu}"
            f"distributed training: {bool(finetuner_args.local_rank != -1)},"
            f" 16-bits training: {finetuner_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {finetuner_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(finetuner_args.output_dir) and finetuner_args.do_train and not finetuner_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(finetuner_args.output_dir)
            if last_checkpoint is None and len(os.listdir(finetuner_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({finetuner_args.output_dir}) already"
                    " exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and finetuner_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at"
                    f" {last_checkpoint}. To avoid this behavior, change"
                    " the `--output_dir` or add `--overwrite_output_dir` to"
                    " train from scratch."
                )
        self.last_checkpoint = last_checkpoint

        # Set seed before initializing model.
        set_seed(finetuner_args.seed)


    def group_text(self, tokenized_datasets, model_max_length):
        """
        Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
        a dictionary.
        """
        data_args = self.data_args
        finetuner_args = self.finetuner_args

        if data_args.block_size is None:
            block_size = model_max_length
            if block_size > 1024:
                logger.warning(
	    			"The chosen tokenizer supports a `model_max_length` that is"
	    			" longer than the default `block_size` value"
	    			" of 1024. If you would like to use a longer `block_size`"
	    			" up to `tokenizer.model_max_length` you can override this "
	    			" default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger"
	    			f" than the maximum length for the model"
                    f"({model_max_length})."
                    f" Using block_size={model_max_length}."
                )
            block_size = min(data_args.block_size, model_max_length)

        # Main data processing function that will concatenate all texts from
        # our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model
            # supported it instead of this drop, you can customize this part to
            # your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts
        # together, so group_texts throws away a remainder for each of those
        # groups of 1,000 texts. You can adjust that batch_size here but a
        # higher value might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation
        # of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        with finetuner_args.main_process_first(desc="grouping texts together"):
            group_batch_size = data_args.group_texts_batch_size
            if data_args.disable_group_texts:
                group_batch_size = 1
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                )

        return lm_datasets


    def tune(self, model, dataset, transform_dataset_in_place=True):
        """
        Perform tuning for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform tuning.
        
        dataset:
            dataset to train model.

        """   
        model_args = self.model_args
        data_args = self.data_args
        finetuner_args = self.finetuner_args
        if not transform_dataset_in_place:
            dataset = copy.deepcopy(dataset)

        # Tokenization and text grouping must be done in the main process
        with finetuner_args.main_process_first(desc="dataset map tokenization"):
            tokenized_dataset = model.tokenize(dataset)
            lm_dataset = self.group_text(
                tokenized_dataset,
                model_max_length=model.get_max_length(),
            )

        train_dataset = lm_dataset.get_backend_dataset()

        if finetuner_args.do_eval:
            eval_dataset_args = deepcopy(data_args)
            eval_dataset_args.dataset_path = finetuner_args.eval_dataset_path
            eval_dataset = Dataset(eval_dataset_args)
            with finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(eval_dataset)
                lm_dataset = self.group_text(
                    tokenized_dataset,
                    model_max_length=model.get_max_length(),
                )
            eval_dataset = lm_dataset.get_backend_dataset()
            

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_preds):
                # import pdb; pdb.set_trace()
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
            
        if finetuner_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        # Initialize our Trainer
        training_args = finetuner_args

        if model_args.use_lora:
            FinetuningTrainer = PeftTrainer
            trainer_callbacks = [PeftSavingCallback]
        else:
            FinetuningTrainer = Trainer
            trainer_callbacks = []

        trainer = FinetuningTrainer(
            model=model.get_backend_model(),
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=model.get_tokenizer(),
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
            callbacks=trainer_callbacks
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            last_checkpoint = self.last_checkpoint
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            if not model_args.use_lora:
                trainer.save_model()  # Saves the tokenizer too for easy upload
            else:
                if model_args.save_aggregated_lora:
                    model.merge_lora_weights()
                model.save(finetuner_args.output_dir,model_args.save_aggregated_lora)

            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

        return model
