#!/usr/bin/env python
# coding=utf-8
"""
The Aligner class simplifies the process of running alignment.
"""

import logging
import numpy as np
import os
import sys
import time
from itertools import chain

import torch
import torch.distributed as dist
import transformers
from datasets import (
    set_caching_enabled,
    Dataset,
    DatasetDict,
)
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
)
from transformers.testing_utils import CaptureLogger

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset as LMFlowDataset
from lmflow.pipeline.base_aligner import BaseAligner
from lmflow.pipeline.utils.raft_trainer import RaftTrainer
import json

logger = logging.getLogger(__name__)


class RaftAligner(BaseAligner):
    """
    Initializes the `RaftAligner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    raft_aligner_args : RaftAlignerArguments object.
        Contains the arguments required to perform alignment.

    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, aligner_args, *args, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = aligner_args

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.INF = 888888888
        logger.setLevel(logging.INFO)


        ##########################
        self.raft_infer_samples_store_dir = aligner_args.raft_infer_set + "/my_infer_set.json"
        self.raft_filter_samples_store_dir = aligner_args.raft_filtered_set + "/my_filtered_set.json"
        self.raft_eval_samples_store_dir = model_args.model_name_or_path + "/eval_set/my_eval_set.json" 
        self.raft_rewards_store_dir = aligner_args.raft_exp_dir + "/reward_record.txt"
        #########################

    def _initialize_trainer(self, model, tokenizer, training_args):
        """
        This function takes the model and tokenizer as the input and initialize the trainer.
        """
        trainer = RaftTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_dict({"text": [ " " ] }),
            eval_dataset=Dataset.from_dict({}),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )
        return trainer


    def _load_input_dataset(self, dataset, tokenizer):
        """
        Load input dataset (i.e. prompt/question dataset) for training.

        Args:
            dataset: A Dataset object.
                The dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        ds = dataset.get_backend_dataset()
        
        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["text"])
            sample['input'] = tokenizer.decode(sample["input_ids"])
            return sample

        if self.mode == "raft_get_rewards":
            pass
        elif self.mode == "raft_get_samples":
            ds = ds.map(tokenize, batched=False)
            ds = ds.filter(lambda x: len(x["input_ids"]) <= 256)
        else:
            raise NotImplementedError("We only support two modes for raft aligner")
        
        ds.set_format(type='torch')

        return ds

    def _clean_text(self, text):
        if len(text) == 0:
            return text
        stext = [x for x in text.split("###Human") if x]
        return stext[0].strip().strip("#") 

    def _discard_sample(self, text):
        if "#" in text:
            return True
        elif len(text) < 2: # delete empty sample
            return True
        return False


    def _get_batch_dataset_top(
        self,
        model,
        batch_input,
        output_min_length=16,
        output_max_length=48,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
    ):
        """
        :param batch_input: input prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()

        querys = batch_input['input']
        data_size = len(querys)

        input_texts = []
        responses = []

        for i, query in enumerate(querys):
            input_texts.append(query)
            if (i + 1) % infer_batch_size == 0 or (i+1 == data_size):
                gen_len = np.random.randint(output_min_length, output_max_length)
                generation_kwargs["max_new_tokens"] = gen_len
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_texts = [
                    generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                ]

                responses.extend(generated_texts)
                input_texts = []

        data = [{"input": querys[j], "output": [responses[j]]} for j in range(data_size)]


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size
        dist.all_gather_object(all_process_list, data)
        gathered_data = []
        for i in range(world_size):
            gathered_data.extend(all_process_list[i])

        if training_args.local_rank == 0:
            logger.info(f"collected data of {len(gathered_data)}")
            output_dataset = {}
            output_dataset['type'] = 'text_only'
            output_dataset['instances'] = gathered_data

            with open(self.raft_infer_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)


    def _get_batch_dataset_local(
            self,
            model,
            batch_input,
            K=8,
            output_min_length=16,
            output_max_length=48,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            reward_model=None,
        ):
            """
            :param batch_input: input prompts
            """
            # we will get the batch dataset via Dataset.from_dict

            querys = batch_input['input']
            data_size = len(querys)

            input_texts = []
            responses = []
            data = []

            for i, query in enumerate(querys):
                input_texts = [query for _ in range(K)]
                
                gen_len = np.random.randint(output_min_length, output_max_length)
                generation_kwargs["max_new_tokens"] = gen_len
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_texts = [
                    generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                ]
                generated_texts = [
                    self._clean_text(generated_text) for generated_text in generated_texts
                ]
               
                data.append({"input": querys[i], "output": [generated_texts[j] for j in range(K)]})
                input_texts = []

            world_size = int(os.getenv("WORLD_SIZE", "1"))
            all_process_list =[{}] * world_size
            dist.all_gather_object(all_process_list, data)
            gathered_data = []
            for i in range(world_size):
                gathered_data.extend(all_process_list[i])

            if training_args.local_rank == 0:
                logger.info(f"collected data of {len(gathered_data)}")
                output_dataset = {}
                output_dataset['type'] = 'text_only'
                output_dataset['instances'] = gathered_data

                with open(self.raft_infer_samples_store_dir, 'w', encoding='utf8') as f:
                    json.dump(output_dataset, f, ensure_ascii=False)


    def _get_reward(
        self,
        batch_input,
        reward_model,
        training_args
    ):
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output']
        data = []
        K = len(responses[0])
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]

            reward_eva.append(rewards[0])

            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -self.INF
                        
            idx_to_record = np.argmax(rewards)
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] != -self.INF:
                data.append({"text": q + tmp_responses[idx_to_record]})
                reward_train.append(rewards[idx_to_record])


        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        data_to_send = [[data[i], reward_train[i]] for i in range(len(data))]
        dist.all_gather_object(all_process_list, data_to_send)
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]]
            gathered_data.extend(tmp_data)

            tmp_train_reward = [tmp[1] for tmp in all_process_list[i]]
            gathered_train_reward.extend(tmp_train_reward)


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        data_to_send = reward_eva
        dist.all_gather_object(all_process_list, data_to_send)
        for i in range(world_size):
            tmp_reward = all_process_list[i]
            gathered_reward.extend(tmp_reward)

        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if training_args.local_rank == 0:
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "\n")


    def align(self, model, dataset, reward_model):
        """
        Perform alignment for a model

        Parameters
        ------------
        model : BaseModel object.
        dataset: Dataset object.
            Input dataset for model to generate outputs. The input and output
                will then be feed into reward model to get the reward for
                alignment.
        reward_model: RegressionModel object.
        """
        tokenizer = model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"


        wrapped_model = model
        model = model.get_backend_model()

        generation_kwargs = {
            "min_length": 1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature":0.85,
        }

        aligner_args = self.aligner_args
        training_args = aligner_args
        model_args = self.model_args
        data_args = self.data_args

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.mode = aligner_args.raft_mode
        self.seed = aligner_args.raft_random_seed
        set_seed(self.seed + training_args.local_rank)
        np.random.seed(self.seed + training_args.local_rank)

        # We split the dataset to each gpu
        dataset = self._load_input_dataset(dataset, tokenizer)
        set_caching_enabled(False)

        share = int(len(dataset) / world_size) 
        dataset = dataset.select(np.arange(training_args.local_rank * share, (training_args.local_rank + 1)*share))


        assert self.mode != "xxx"
        if self.mode == "raft_get_samples":
            collection_strategy = aligner_args.collection_strategy
            sft_batch_size = aligner_args.raft_batch_size
            if collection_strategy == "top":
                alpha = aligner_args.top_reward_percentage
                M = int(sft_batch_size / world_size / alpha) 
            elif collection_strategy == "local":
                K = int(1/aligner_args.top_reward_percentage)
                M = int(sft_batch_size / world_size)
            else:
                raise NotImplementedError("We only support two data collection strategies")

            # we collect samples by the current model
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            # to use deepspeed...
            raft_trainer = self._initialize_trainer(model, tokenizer, training_args)
            raft_trainer.train(resume_from_checkpoint=False, is_first_time=True)
            data_size = len(shuffled_dataset)
            random_idxs = np.arange(data_size)
            np.random.shuffle(random_idxs)
            end_idx = np.min([data_size, M])
            batch_input = shuffled_dataset.select(random_idxs[0 : end_idx])

            model.gradient_checkpointing_disable()
            model.config.use_cache = True

            start_time = time.time()
            if collection_strategy == "top":
                self._get_batch_dataset_top(
                        raft_trainer.tmp_model,
                        batch_input,
                        output_min_length=aligner_args.output_min_length,
                        output_max_length=aligner_args.output_max_length,
                        generation_kwargs=generation_kwargs,
                        tokenizer=tokenizer,
                        training_args=training_args,
                        reward_model=reward_model)
            elif collection_strategy == "local":
                self._get_batch_dataset_local(
                        raft_trainer.tmp_model,
                        batch_input, 
                        K, 
                        output_min_length=aligner_args.output_min_length,
                        output_max_length=aligner_args.output_max_length,
                        generation_kwargs=generation_kwargs,
                        tokenizer=tokenizer,
                        training_args=training_args,
                        reward_model=reward_model)
            end_time = time.time()
            logger.info("It takes %.2f s to inference one stage", end_time - start_time)

        elif self.mode == "raft_get_rewards":
            batch_input = dataset
            start_time = time.time()
            self._get_reward(
                batch_input, 
                reward_model=reward_model,
                training_args=training_args
            )
            end_time = time.time()
            logger.info("It takes %.2f s to inference the rewards", end_time - start_time)

        return wrapped_model 
