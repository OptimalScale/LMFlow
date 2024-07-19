import copy
import importlib.resources as pkg_resources
import logging
import os
import subprocess
import sys
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from tqdm import tqdm
import torch
from transformers import TrainingArguments

from lmflow.pipeline.utils.dpov2_trainer import DPOv2Trainer
from lmflow.pipeline.base_aligner import BaseAligner
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    DPOv2AlignerArguments
)
from lmflow.utils.common import (
    make_shell_args_from_dataclass, 
    add_dataclass_attr_prefix, 
    create_copied_dataclass
)
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.datasets.dataset import Dataset, KEY_SCORE, KEY_TYPE, KEY_INSTANCES
from lmflow.utils.constants import MEMORY_SAFE_DPOV2_ALIGN_ENV_VAR_TO_REMOVE


logger = logging.getLogger(__name__)


ReferenceModelArguments = create_copied_dataclass(
    original_dataclass=ModelArguments, 
    field_prefix="reference_",
    class_prefix="Reference"
)


class DPOv2Aligner(BaseAligner):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        aligner_args: DPOv2AlignerArguments,
        ref_model_args: ModelArguments,
    ):
        self.model_args = model_args
        self.ref_model_args = ref_model_args
        self.data_args = data_args
        self.aligner_args = aligner_args


    def align(
        self,
        model: HFDecoderModel,
        ref_model: HFDecoderModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        transform_dataset_in_place: bool=True,
    ):
        if (train_dataset.get_type() not in ["text_to_scored_textlist", "paired_text_to_text"]) or \
            (eval_dataset.get_type() not in ["text_to_scored_textlist", "paired_text_to_text"]):
            raise ValueError(
                f"Unsupported dataset type {train_dataset.get_type()} for DPOv2 aligner."
            )
            
        # step 0. setting up
        if self.aligner_args.gradient_checkpointing:
            logger.warning(
                "Setting backend_model.config.use_cache to False since using gradient checkpointing"
            )
            model.get_backend_model().config.use_cache = False
            ref_model.get_backend_model().config.use_cache = False
            
        # step 1. prepare datasets
        if train_dataset.get_type() == "text_to_scored_textlist":
            train_dataset = self.convert_to_paired_dataset(
                source_dataset=train_dataset,
                sampling_paired_method=self.aligner_args.sampling_paired_method,
                length_penalty=self.aligner_args.length_penalty,
                margin_scale=self.aligner_args.margin_scale,
                use_fast=False,
            )
        if self.data_args.max_train_samples:
            train_dataset.backend_dataset = train_dataset.backend_dataset.select(range(self.data_args.max_train_samples))

        if eval_dataset.get_type() == "text_to_scored_textlist":
            eval_dataset = self.convert_to_paired_dataset(
                source_dataset=eval_dataset,
                sampling_paired_method=self.aligner_args.sampling_paired_method,
                margin_scale=self.aligner_args.margin_scale,
                use_fast=False,
            )
            
        # step 2. prepare trainer
        dpo_trainer = DPOv2Trainer(
            model.get_backend_model(),
            ref_model.get_backend_model(),
            train_dataset=train_dataset.get_backend_dataset(), # tokenization is done in the trainer
            eval_dataset=eval_dataset.get_backend_dataset(),
            tokenizer=model.tokenizer,
            args=self.__prepare_training_args(self.aligner_args),
            beta=self.aligner_args.beta,
            loss_type=self.aligner_args.loss_type,
            max_prompt_length=self.aligner_args.max_prompt_length,
            max_length=self.aligner_args.max_length,
            mask_prompt=self.aligner_args.mask_prompt,
            len_penalty=self.aligner_args.length_penalty,
            # preprocessing_num_workers=self.data_args.preprocessing_num_workers, # will trigger TypeError: cannot pickle 'torch._C._distributed_c10d.ProcessGroup' object
        )
        
        # step 3. train
        dpo_trainer.train()
        dpo_trainer.save_model(self.aligner_args.output_dir)

        # step 4. save
        output_dir = os.path.join(self.aligner_args.output_dir, "final_checkpoint")
        dpo_trainer.model.save_pretrained(output_dir)
        
        # step 5. release resources
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        
    def __prepare_training_args(
        self,
        args: DPOv2AlignerArguments,
    ) -> TrainingArguments:
        training_args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            save_strategy=args.save_strategy,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            learning_rate=args.learning_rate,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            output_dir=args.output_dir,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            optim=args.optim,
            bf16=args.bf16,
            report_to=args.report_to,
            run_name=args.run_name,
            remove_unused_columns=False, # DO NOT CHANGE THIS, may cause error https://discuss.huggingface.co/t/indexerror-invalid-key-16-is-out-of-bounds-for-size-0/14298/3
        )
        logger.debug(f"Actual training arguments for dpo trainer: {training_args}")
        
        return training_args
    
    
    def convert_to_paired_dataset(
        self,
        source_dataset: Dataset,
        sampling_paired_method: str="random",
        length_penalty: float=0.0,
        margin_scale: float=1.0,
        use_fast: bool=False,
    ) -> Dataset:
        """Convert a scored one to multiple (text_to_scored_textlist) to a paired dataset by rejection sampling.
        """
        output_dict = {
            KEY_INSTANCES: []
        }
        if source_dataset.get_type() in ["text_to_scored_textlist"]:
            output_dict[KEY_TYPE] = "paired_text_to_text"
        
        for sample in tqdm(source_dataset.get_backend_dataset(), desc="Converting to paired dataset"):
            sample_output_dict = {}
            lengths = self._calc_response_lengths(sample["output"], source_dataset.get_type())
            penalized_rewards = self._calc_reward_with_length_penalty(
                rewards=[content[KEY_SCORE] for content in sample["output"]], 
                lengths=lengths, 
                length_penalty=length_penalty
            )
            chosen_idx, rejected_idx = self.sampling_paired_idx_from_rewards(
                rewards=penalized_rewards, 
                sampling_paired_method=sampling_paired_method,
                use_fast=use_fast
            )
            
            sample_output_dict["prompt"] = sample["input"]
            sample_output_dict["chosen"] = sample["output"][chosen_idx]["text"]
            sample_output_dict["rejected"] = sample["output"][rejected_idx]["text"]
            sample_output_dict["margin"] = (sample["output"][chosen_idx][KEY_SCORE] - sample["output"][rejected_idx][KEY_SCORE]) * margin_scale
            output_dict[KEY_INSTANCES].append(sample_output_dict)
        
        output_dataset_args = copy.deepcopy(source_dataset.data_args)
        output_dataset_args.dataset_path = None
        output_dataset_args.dataset_name = f"paired_{output_dataset_args.dataset_name}"
        output_dataset = Dataset(output_dataset_args)
        output_dataset = output_dataset.from_dict(output_dict)
        
        return output_dataset
    
    
    def _calc_response_lengths(
        self,
        outputs: List[Union[str, Dict[str, str]]],
        dataset_type: str,
    ) -> List[int]:
        all_lengths = []
        if dataset_type == "text_to_scored_textlist":
            all_lengths = [len(output["text"]) for output in outputs]
            
        else:
            raise NotImplementedError(
                f"Unknown dataset type {dataset_type} when calculating the response length."
            )
        
        return all_lengths
    
    
    def _calc_reward_with_length_penalty(
        self,
        rewards: List[float], 
        lengths: List[int], 
        length_penalty: float,
    ) -> List[float]:
        """When length_penalty > 0, penalize the longer sequence by subtracting 
        length_penalty * length from the reward. Vice versa when length_penalty < 0.
        """
        assert len(rewards) == len(lengths), "The number of rewards and lengths should be the same."
        return [reward - length_penalty * length for reward, length in zip(rewards, lengths)]
    
    
    def sampling_paired_idx_from_rewards(
        self,
        rewards: List[float],
        sampling_paired_method: str="random",
        use_fast: bool=False,
    ) -> Tuple[int, int]:
        """Prepare the dataset for DPO training by rejection sampling.
        We implement different strategies to select pairs, including
        random: randomly select two instances
        max_min: best v.s. worst
        max_max: best v.s. second best
        max_random: best v.s. random from the remaining
        """
        if use_fast:
            return self._sampling_paired_idx_from_rewards_fast(rewards, sampling_paired_method)
        else:
            return self._sampling_paired_idx_from_rewards(rewards, sampling_paired_method)


    def _sampling_paired_idx_from_rewards(
        self,
        rewards: List[float], 
        sampling_paired_method: str="random"
    ) -> Tuple[int, int]:
        idx_0, idx_1 = -1, -1
        
        if sampling_paired_method == "random":
            idx_0, idx_1 = np.random.choice(len(rewards), size=2, replace=False)
        elif sampling_paired_method == "max_min":
            idx_0, idx_1 = np.argmax(rewards), np.argmin(rewards)
        elif sampling_paired_method == "max_max":
            sorted_indices = np.argsort(rewards)
            idx_0, idx_1 = sorted_indices[-1], sorted_indices[-2]
        elif sampling_paired_method == "max_random":
            idx_0 = np.argmax(rewards)
            idx_1 = np.random.choice([i for i in range(len(rewards)) if i != idx_0])
        else:
            raise ValueError(f"Unknown sampling method: {sampling_paired_method}")
        
        chosen_idx, rejected_idx = (idx_0, idx_1) if rewards[idx_0] > rewards[idx_1] else (idx_1, idx_0)
        
        return chosen_idx, rejected_idx


    def _sampling_paired_idx_from_rewards_fast(
        self,
        rewards: List[float],
        sampling_paired_method: str="random"
    ) -> Tuple[int, int]:
        idx_0, idx_1 = -1, -1
        
        if sampling_paired_method == "random":
            idx_0, idx_1 = 0, 1
        elif sampling_paired_method == "max_min":
            idx_0, idx_1 = np.argmax(rewards), np.argmin(rewards)
        elif sampling_paired_method == "max_max":
            sorted_indices = np.argsort(rewards)
            idx_0, idx_1 = sorted_indices[-1], sorted_indices[-2]
        elif sampling_paired_method == "max_random":
            idx_0 = np.argmax(rewards)
            idx_1 = 0 if idx_0 != 0 else 1
        else:
            raise ValueError(f"Unknown sampling method: {sampling_paired_method}")
        
        chosen_idx, rejected_idx = (idx_0, idx_1) if rewards[idx_0] > rewards[idx_1] else (idx_1, idx_0)
        
        return chosen_idx, rejected_idx
        

class MemorySafeDPOv2Aligner:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        aligner_args: DPOv2AlignerArguments,
        ref_model_args: ModelArguments,
    ):
        self.model_args = model_args
        self.ref_model_args = ReferenceModelArguments(**add_dataclass_attr_prefix(ref_model_args, 'reference_'))
        self.data_args = data_args
        self.aligner_args = aligner_args
        self.aligner_file_path = pkg_resources.files("lmflow.pipeline.utils") / "memory_safe_dpov2_align.py"

    def align(self):
        aligner_args = make_shell_args_from_dataclass(
            dataclass_objects=[
                self.model_args, 
                self.data_args, 
                self.aligner_args,
                self.ref_model_args
            ],
            format="shell",
            ignored_args_list=['accelerator_config', 'fsdp_config', '_n_gpu'],
        )
        cmd = (
            f"accelerate launch --config_file {self.aligner_args.accelerate_config_file}"
            + " " 
            + str(self.aligner_file_path) 
            + " " 
            + aligner_args
        )
        current_env = os.environ.copy()
        for var in MEMORY_SAFE_DPOV2_ALIGN_ENV_VAR_TO_REMOVE:
            current_env.pop(var, None)
        
        cli_res = subprocess.run(
            args=cmd,
            stdout=sys.stdout,
            stderr=sys.stdout,
            shell=True,
            preexec_fn=os.setsid,
            env=current_env,
        )
        logger.info(f"MemorySafeDPOv2Aligner subprocess run finished, info at finish: {cli_res}")
        
        if cli_res.returncode != 0:
            print(cli_res.stderr)
            raise RuntimeError(f"Error during MemorySafeDPOv2Aligner: {cli_res}")