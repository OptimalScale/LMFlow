import sys
import logging

import torch
import torch.nn as nn
import datasets
import transformers
from transformers import (
    set_seed,
    Trainer
)
from transformers.utils import send_example_telemetry

from lmflow.pipeline.base_tuner import BaseTuner


logger = logging.getLogger(__name__)


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class RewardModelingTuner(BaseTuner):
    """Initializes the `RewardModelingTuner` class.

    Parameters
    ----------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    rmtuner_args : RewardModelingArguments object.
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
        rmtuner_args, 
        *args, 
        **kwargs
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.rmtuner_args = rmtuner_args
        
        
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

        log_level = rmtuner_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {rmtuner_args.local_rank},"
            f" device: {rmtuner_args.device},"
            f" n_gpu: {rmtuner_args.n_gpu},"
            f" distributed training: {bool(rmtuner_args.local_rank != -1)},"
            f" 16-bits training: {rmtuner_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {rmtuner_args}")

        # Set seed before initializing model.
        set_seed(rmtuner_args.seed)
        
    
    def tune(
        self,
        model,
        data_collator=None
    ):
        if self.rmtuner_args.do_train:
            trainer = RewardTrainer(
                model=model,
                args=self.rmtuner_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
            )
                
            train_result = trainer.train()
            
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            trainer.save_model()
            tokenizer.save_pretrained(output_name)