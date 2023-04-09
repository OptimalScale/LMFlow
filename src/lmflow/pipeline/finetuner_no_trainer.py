# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import json
import logging
import math
import os
import sys


import datasets
import transformers

from itertools import chain
import torch
from tqdm.auto import tqdm
from transformers import (
    default_data_collator,
    get_scheduler
)
from transformers.utils import send_example_telemetry
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType
# from accelerate.logging import get_logger
from accelerate.utils import set_seed
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_tuner import BaseTuner


logger = logging.getLogger(__name__)


class Finetuner_no_trainer(BaseTuner):
    """
    Initializes the `Finetune_no_tuner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_no_trainer_args : Finetuner Arguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, finetuner_no_trainer_args, *args, **kwargs):
        
        self.model_args = model_args
        self.data_args = data_args
        self.finetuner_no_trainer_args = finetuner_no_trainer_args

        # Sending telemetry. Tracking the example usage helps us better
        # allocate resources to maintain them. The information sent is the one
        # passed as arguments along with your Python/PyTorch versions.
        send_example_telemetry("run_clm_no_trainer", model_args, data_args)

        self.accelerator = Accelerator()

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # logger.info(self.accelerator.state, main_process_only=False)
        # if self.accelerator.is_local_main_process:
        #     datasets.utils.logging.set_verbosity_warning()
        #     transformers.utils.logging.set_verbosity_info()
        # else:
        #     datasets.utils.logging.set_verbosity_error()
        #     transformers.utils.logging.set_verbosity_error()

        log_level = finetuner_no_trainer_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {finetuner_no_trainer_args.local_rank},"
            f" device: {finetuner_no_trainer_args.device},"
            f" n_gpu: {finetuner_no_trainer_args.n_gpu}"
            f"distributed training: {bool(finetuner_no_trainer_args.local_rank != -1)},"
            f" 16-bits training: {finetuner_no_trainer_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {finetuner_no_trainer_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(finetuner_no_trainer_args.output_dir) and finetuner_no_trainer_args.do_train and not finetuner_no_trainer_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(finetuner_no_trainer_args.output_dir)
            if last_checkpoint is None and len(os.listdir(finetuner_no_trainer_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({finetuner_no_trainer_args.output_dir}) already"
                    " exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and finetuner_no_trainer_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at"
                    f" {last_checkpoint}. To avoid this behavior, change"
                    " the `--output_dir` or add `--overwrite_output_dir` to"
                    " train from scratch."
                )
        self.last_checkpoint = last_checkpoint

        # Set seed before initializing model.
        set_seed(finetuner_no_trainer_args.seed)
        
        self.accelerator.wait_for_everyone()

    def group_text(self, tokenized_datasets, model_max_length):
        """
        Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
        a dictionary.
        """
        data_args = self.data_args
        finetuner_no_trainer_args = self.finetuner_no_trainer_args

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
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts
        # together, so group_texts throws away a remainder for each of those
        # groups of 1,000 texts. You can adjust that batch_size here but a
        # higher value might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation
        # of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        with self.accelerator.main_process_first():
            group_batch_size = 1000
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


    def tune(self, model, lm_dataset):
        """
        Perform tuning for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform tuning.
        
        lm_dataset:
            dataset to train model.

        """   
        model_args = self.model_args
        data_args = self.data_args
        finetuner_no_trainer_args = self.finetuner_no_trainer_args

        finetuner_no_trainer_args.checkpointing_steps = None
        finetuner_no_trainer_args.max_train_steps = None

        finetuner_no_trainer_args.with_tracking = True

        train_dataset = lm_dataset.get_backend_dataset()


        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=finetuner_no_trainer_args.per_device_train_batch_size
        )

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.get_backend_model().named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": finetuner_no_trainer_args.weight_decay,
            },
            {
                "params": [p for n, p in model.get_backend_model().named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=finetuner_no_trainer_args.learning_rate)

        if finetuner_no_trainer_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
        

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / finetuner_no_trainer_args.gradient_accumulation_steps)
        if finetuner_no_trainer_args.max_train_steps is None:
            finetuner_no_trainer_args.max_train_steps = int(finetuner_no_trainer_args.num_train_epochs * num_update_steps_per_epoch)
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=finetuner_no_trainer_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=finetuner_no_trainer_args.warmup_steps * finetuner_no_trainer_args.gradient_accumulation_steps,
            num_training_steps=finetuner_no_trainer_args.max_train_steps * finetuner_no_trainer_args.gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            model.get_backend_model(), optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / finetuner_no_trainer_args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            finetuner_no_trainer_args.max_train_steps = int(finetuner_no_trainer_args.num_train_epochs * num_update_steps_per_epoch)
        # Afterwards we recalculate our number of training epochs
        finetuner_no_trainer_args.num_train_epochs = math.ceil(finetuner_no_trainer_args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = finetuner_no_trainer_args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if finetuner_no_trainer_args.with_tracking:
            experiment_config = vars(finetuner_no_trainer_args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            self.accelerator.init_trackers("clm_no_trainer", experiment_config)

        # Train!
        total_batch_size = finetuner_no_trainer_args.per_device_train_batch_size * self.accelerator.num_processes * finetuner_no_trainer_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {finetuner_no_trainer_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {finetuner_no_trainer_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {finetuner_no_trainer_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {finetuner_no_trainer_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(finetuner_no_trainer_args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if finetuner_no_trainer_args.resume_from_checkpoint:
            if finetuner_no_trainer_args.resume_from_checkpoint is not None or finetuner_no_trainer_args.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {finetuner_no_trainer_args.resume_from_checkpoint}")
                self.accelerator.load_state(finetuner_no_trainer_args.resume_from_checkpoint)
                path = os.path.basename(finetuner_no_trainer_args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * finetuner_no_trainer_args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)


        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch

        for epoch in range(starting_epoch, finetuner_no_trainer_args.num_train_epochs):
            model.train()
            if finetuner_no_trainer_args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if finetuner_no_trainer_args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % finetuner_no_trainer_args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue

                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if finetuner_no_trainer_args.with_tracking:
                        total_loss += loss.detach().float()
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if finetuner_no_trainer_args.output_dir is not None:
                            output_dir = os.path.join(finetuner_no_trainer_args.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)
                if completed_steps >= finetuner_no_trainer_args.max_train_steps:
                    break

            starting_step = 0
            
            # model.eval()
            # losses = []
            # for step, batch in enumerate(eval_dataloader):
            #     with torch.no_grad():
            #         outputs = model(**batch)

            #     loss = outputs.loss
            #     losses.append(self.accelerator.gather_for_metrics(loss.repeat(finetuner_no_trainer_args.per_device_eval_batch_size)))

            # losses = torch.cat(losses)
            # try:
            #     eval_loss = torch.mean(losses)
            #     perplexity = math.exp(eval_loss)
            # except OverflowError:
            #     perplexity = float("inf")

            logger.info(f"epoch {epoch}: train_loss: {total_loss.item() / len(train_dataloader)}")
            print(f"epoch {epoch}: train_loss: {total_loss.item() / len(train_dataloader)}")

            if finetuner_no_trainer_args.with_tracking:
                self.accelerator.log(
                    {
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )


            if finetuner_no_trainer_args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if finetuner_no_trainer_args.output_dir is not None:
                    output_dir = os.path.join(finetuner_no_trainer_args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

        

        if finetuner_no_trainer_args.with_tracking:
            self.accelerator.end_training()

        if finetuner_no_trainer_args.output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                finetuner_no_trainer_args.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )


        return model
