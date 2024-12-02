#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset.
"""

import copy
import logging
import os
import sys
from typing import Any, Iterable, Optional, Tuple

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
from transformers import PreTrainedModel, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    send_example_telemetry,
)
import numpy as np

import lmflow.optim.optimizers as optim
from lmflow.args import OptimizerNames
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
            f" n_gpu: {finetuner_args.n_gpu},"
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
                if self.model_args.truncate_to_model_max_length:
                    logger.warning(
                        f"The block_size passed ({data_args.block_size}) is larger"
                        f" than the maximum length for the model"
                        f"({model_max_length})."
                        f" Using block_size={model_max_length}."
                        f"If you would like to use a longer 'block_size' that is"
                        f" longer than the maximum length supported by the model,"
                        f" you can override this behavior with"
                        f"default with `--truncate_to_model_max_length False`."
                    )
                    block_size = model_max_length
                else:
                    logger.warning(
                        f"The block_size passed ({data_args.block_size}) is larger"
                        f"than the maximum length for the model"
                        f"({model_max_length})."
                        f"Using block_size={data_args.block_size}.")
                    block_size = data_args.block_size
            else:
                block_size = data_args.block_size
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

    def create_customized_optimizer(self, base_trainer_class, model_args):
        class CustomizedOptimTrainer(base_trainer_class):

            @staticmethod
            def get_optimizer_cls_and_kwargs(
                args: TrainingArguments,
                model: Optional[PreTrainedModel] = None,
            ) -> Tuple[Any, Any]:
                # parse args.optim_args
                optim_args = {}
                if args.customized_optim_args:
                    for mapping in args.customized_optim_args.replace(" ", "").split(","):
                        key, value = mapping.split("=")
                        optim_args[key] = value

                optimizer_kwargs = {"lr": args.learning_rate}

                if args.customized_optim == OptimizerNames.DUMMY:
                    optimizer_cls = optim.Dummy
                    dummy_kwargs = {
                        "betas": (args.optim_dummy_beta1, args.optim_dummy_beta2),
                    }
                    optimizer_kwargs.update(dummy_kwargs)
                elif args.customized_optim == OptimizerNames.ADABELIEF:
                    optimizer_cls = optim.AdaBelief
                    adabelief_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay)
                    }
                    optimizer_kwargs.update(adabelief_kwargs)
                elif args.customized_optim == OptimizerNames.ADABOUND:
                    optimizer_cls = optim.AdaBound
                    adabound_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay)
                    }
                    optimizer_kwargs.update(adabound_kwargs)
                elif args.customized_optim == OptimizerNames.LARS:
                    optimizer_cls = optim.LARS
                    lars_kwargs = {
                        "momentum": (args.optim_momentum),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(lars_kwargs)
                elif args.customized_optim == OptimizerNames.LAMB:
                    optimizer_cls = optim.Lamb
                    lamb_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(lamb_kwargs)
                elif args.customized_optim == OptimizerNames.ADAMAX:
                    optimizer_cls = optim.Adamax
                    adamax_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(adamax_kwargs)
                elif args.customized_optim == OptimizerNames.NADAM:
                    optimizer_cls = optim.NAdam
                    nadam_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(nadam_kwargs)
                elif args.customized_optim == OptimizerNames.RADAM:
                    optimizer_cls = optim.RAdam
                    radam_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(radam_kwargs)
                elif args.customized_optim == OptimizerNames.ADAMP:
                    optimizer_cls = optim.AdamP
                    adamp_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(adamp_kwargs)
                elif args.customized_optim == OptimizerNames.SGDP:
                    optimizer_cls = optim.SGDP
                    sgdp_kwargs = {
                        "momentum": (args.optim_momentum),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(sgdp_kwargs)
                elif args.customized_optim == OptimizerNames.YOGI:
                    optimizer_cls = optim.Yogi
                    yogi_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(yogi_kwargs)
                elif args.customized_optim == OptimizerNames.SOPHIA:
                    optimizer_cls = optim.SophiaG
                    sophia_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(sophia_kwargs)
                elif args.customized_optim == OptimizerNames.ADAM:
                    optimizer_cls = optim.Adam
                    adam_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                    }
                    optimizer_kwargs.update(adam_kwargs)
                elif args.customized_optim == OptimizerNames.NOVOGRAD:
                    optimizer_cls = optim.NovoGrad
                    novograd_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(novograd_kwargs)
                elif args.customized_optim == OptimizerNames.ADADELTA:
                    optimizer_cls = optim.Adadelta
                    adadelta_kwargs = {
                    }
                    optimizer_kwargs.update(adadelta_kwargs)
                elif args.customized_optim == OptimizerNames.ADAGRAD:
                    optimizer_cls = optim.AdaGrad
                    adagrad_kwargs = {
                    }
                    optimizer_kwargs.update(adagrad_kwargs)
                elif args.customized_optim == OptimizerNames.ADAMW_SCHEDULE_FREE:
                    optimizer_cls = optim.AdamWScheduleFree
                    adamw_schedule_free_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(adamw_schedule_free_kwargs)
                elif args.customized_optim == OptimizerNames.SGD_SCHEDULE_FREE:
                    optimizer_cls = optim.SGDScheduleFree
                    sgd_schedule_free_kwargs = {
                        "momentum": (args.optim_momentum),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(sgd_schedule_free_kwargs)
                elif args.customized_optim == OptimizerNames.ADAN:
                    optimizer_cls = optim.Adan
                    adan_kwargs = {
                        "betas": (args.optim_beta1, args.optim_beta2, args.optim_beta3),
                        "weight_decay": (args.optim_weight_decay),
                    }
                    optimizer_kwargs.update(adan_kwargs)
                else:
                    raise ValueError(
                        f"Trainer cannot instantiate unsupported optimizer: "
                        f" {args.customized_optim}"
                    )
                return optimizer_cls, optimizer_kwargs

            def create_optimizer(self):
                opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

                if self.optimizer is None:
                    decay_parameters = self.get_decay_parameter_names(opt_model)
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters()
                                    if (n in decay_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters()
                                    if (n not in decay_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]

                    optimizer_cls, optimizer_kwargs = CustomizedOptimTrainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

                    # Overwrite `params` in case it's created by
                    # `get_optimizer_cls_and_kwargs` e.g. for GaLore optimizer.
                    if "params" in optimizer_kwargs:
                        optimizer_grouped_parameters = optimizer_kwargs.pop(
                            "params"
                        )

                    # For layer-wise dummy optimizers we overwrite
                    # optimizer_grouped_parameters with `optimizer_dict` to
                    # avoid arguments conflicts.
                    if "optimizer_dict" in optimizer_kwargs:
                        optimizer_grouped_parameters = optimizer_kwargs.pop(
                            "optimizer_dict"
                        )

                    self.optimizer = optimizer_cls(
                        optimizer_grouped_parameters,
                        **optimizer_kwargs
                    )
                if is_sagemaker_mp_enabled():
                    self.optimizer = smp.DistributedOptimizer(self.optimizer)
                    
        return CustomizedOptimTrainer

    def tune(self,
             model,
             dataset,
             transform_dataset_in_place=True,
             data_collator=None):
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
        if dataset.backend == "custom_multi_modal":
            dataset.backend_dataset.register_tokenizer(
                model.tokenizer, model.image_processor)
            lm_dataset = dataset
        else:
            with finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(dataset)
                if data_args.disable_group_texts:
                    lm_dataset = tokenized_dataset
                else:
                    lm_dataset = self.group_text(
                        tokenized_dataset,
                        model_max_length=model.get_max_length(),
                    )

        train_dataset = lm_dataset.get_backend_dataset()
        logger.info(f"Number of train samples: {len(train_dataset)}")

        if finetuner_args.do_eval:
            eval_dataset_args = deepcopy(data_args)
            eval_dataset_args.dataset_path = finetuner_args.eval_dataset_path
            eval_dataset = Dataset(eval_dataset_args)
            with finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(eval_dataset)
                if data_args.disable_group_texts:
                    lm_dataset = tokenized_dataset
                else:
                    lm_dataset = self.group_text(
                        tokenized_dataset,
                        model_max_length=model.get_max_length(),
                    )
            eval_dataset = lm_dataset.get_backend_dataset()
            logger.info(f"Number of eval samples: {len(eval_dataset)}")

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_preds):
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
        if data_collator is None:
            data_collator = default_data_collator

        if training_args.use_customized_optim:
            BaseTrainer = FinetuningTrainer
            FinetuningTrainer = self.create_customized_optimizer(
                BaseTrainer, model_args
            )

        if training_args.use_lisa:
            class DynamicLayerActivationCallback(TrainerCallback):
                def __init__(self, n_layers, interval_steps, model):
                    super().__init__()
                    self.n_layers = n_layers
                    self.interval_steps = interval_steps
                    self.model = model

                    # Determine the way to access layers based on the model type
                    class_to_layers_map = {
                        'LlamaForCausalLM': 'model.model.layers',
                        'Qwen2ForCausalLM': 'model.model.layers',
                        'MistralForCausalLM': 'model.model.layers',
                        'MixtralForCausalLM': 'model.model.layers',
                        'GemmaForCausalLM': 'model.model.layers',
                        'GPT2LMHeadModel': 'model.transformer.h',
                        'HymbaForCausalLM': 'model.model.layers',
                    }
                    model_class_name = self.model.__class__.__name__
                    if model_class_name in class_to_layers_map:
                        self.layers_attribute = class_to_layers_map[model_class_name]
                    else:
                        self.layers_attribute = training_args.lisa_layers_attribute
                    self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers

                    self.active_layers_indices = []

                def freeze_all_layers(self):
                    layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
                    for layer in layers:
                        for param in layer.parameters():
                            param.requires_grad = False

                def on_step_begin(self, args, state, control, **kwargs):
                    # Check if it's time to switch active layers, including at step 0
                    if state.global_step % self.interval_steps == 0:
                        self.switch_active_layers()

                def switch_active_layers(self):
                    # First, disable gradients for all layers
                    self.freeze_all_layers()

                     # Randomly select n_layers to activate
                    layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
                    self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
                    print(f"Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

                    # Enable gradients only for the selected layers
                    for idx in self.active_layers_indices:
                        for param in layers[idx].parameters():
                            param.requires_grad = True

            # Instantiate the callback
            dynamic_layer_activation_callback = DynamicLayerActivationCallback(
                n_layers=training_args.lisa_activated_layers,               # Number of layers to activate
                interval_steps=training_args.lisa_interval_steps,           # Step interval to update active layers
                model=model.get_backend_model()
            )

            trainer_callbacks.append(dynamic_layer_activation_callback)

        trainer = FinetuningTrainer(
            model=model.get_backend_model(),
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=model.get_tokenizer(),
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=data_collator,
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
                model.save(finetuner_args.output_dir, model_args.save_aggregated_lora)
            # save language_projection for multi-modal model;
            if self.finetuner_args.save_language_projection:
                language_projection_state = trainer.model.language_projection.state_dict()
                torch.save(
                    osp.join(
                        self.finetuner_args.output_dir,
                        "language_projection.pth"),
                    language_projection_state)
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