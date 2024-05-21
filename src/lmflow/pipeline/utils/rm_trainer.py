import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer

from .peft_trainer import PeftTrainer


def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result


def rm_loss(model, inputs, return_outputs=False):
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


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return rm_loss(model, inputs, return_outputs)


class PeftRewardTrainer(PeftTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return rm_loss(model, inputs, return_outputs)