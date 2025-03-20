import logging
from typing import Optional, Union, Dict, List, Any, Tuple, Callable, Literal

from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    DataCollator, 
    TrainingArguments, 
    TrainerCallback
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from lmflow.pipeline.utils.dpov2_dataprocessor import PreferenceDataCollatorWithPadding
from lmflow.utils.versioning import is_trl_available

if is_trl_available():
    from trl import DPOTrainer
else:
    raise ImportError("Please install trl package to use dpo_aligner.py")


logger = logging.getLogger(__name__)


class DPOv2Trainer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "cross_entropy", "kl", "rev_kl", "raft"] = "rev_kl",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        mask_prompt: Optional[bool] = False,
        len_penalty: float = 0,
        preprocessing_num_workers: int = 1,
    ):

        if data_collator is None:
            data_collator = PreferenceDataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=False,
                max_target_length=max_target_length,
                mask_prompt=mask_prompt,
            )
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            dataset_num_proc=preprocessing_num_workers,
        )
        self.use_dpo_data_collator = True
        self.len_penalty = len_penalty

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        margin: Optional[torch.FloatTensor] = None,
        len_penalty: float = 0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps + len_penalty

        if reference_free:
            ref_logratios = 0

        if self.loss_type == "sigmoid":
            logits = pi_logratios - ref_logratios
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            logits = pi_logratios - ref_logratios
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "cross_entropy":
            logits = policy_chosen_logps - reference_chosen_logps
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "raft":
            losses = -policy_chosen_logps  # F.logsigmoid(self.beta * logits)
        elif self.loss_type == "ipo":
            logits = pi_logratios - ref_logratios
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kl":
            logits = pi_logratios - ref_logratios
            p = F.sigmoid(self.beta * logits)
            p = torch.minimum(p, torch.ones_like(p) * 0.999)
            p_gt = torch.exp(margin) / (1 + torch.exp(margin) + 1e-3)
            losses = p * (torch.log(p) - torch.log(p_gt)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - p_gt))
        elif self.loss_type == "tv":
            logits = pi_logratios - ref_logratios
            p = F.sigmoid(self.beta * logits)
            p_gt = torch.exp(margin) / (1 + torch.exp(margin))
            losses = torch.abs(p - p_gt)
        elif self.loss_type == "hellinger":
            logits = pi_logratios - ref_logratios
            p = F.sigmoid(self.beta * logits)
            p = torch.minimum(p, torch.ones_like(p) * 0.999)
            p_gt = torch.exp(margin) / (1 + torch.exp(margin))
            losses = 0.5 * ((p**0.5 - p_gt**0.5) ** 2 + ((1 - p) ** 0.5 - (1 - p_gt) ** 0.5) ** 2)
        elif self.loss_type == "rev_kl":
            logits = pi_logratios - ref_logratios
            logp = F.logsigmoid(self.beta * logits)
            logp_neg = F.logsigmoid(-self.beta * logits)
            p_gt = F.sigmoid(margin)
            losses = -p_gt * (logp) - (1 - p_gt) * logp_neg
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        return self.get_batch_metrics(model, batch, train_eval)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)
        if self.len_penalty > 0:
            chosen_len = batch["chosen_input_ids"].shape[1] * self.len_penalty
            rejected_len = batch["rejected_input_ids"].shape[1] * self.len_penalty
            len_penalty = chosen_len - rejected_len
        else:
            chosen_len = 1
            rejected_len = 1
            len_penalty = 0

        margin = torch.tensor(batch["margin"], dtype=policy_chosen_logps.dtype).to(self.accelerator.device)
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            margin=margin,
            len_penalty=len_penalty,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics
