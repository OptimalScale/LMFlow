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

        output_reward_path = aligner_args.output_reward_path
        if output_reward_path is not None:
            os.makedirs(os.path.dirname(output_reward_path), exist_ok=True)
            # Deletes a maybe-exist file
            try:
                os.remove(output_reward_path)
            except OSError:
                pass


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


    def _load_dataset(
        self,
        selected_dataset,
        model,
        tokenizer,
        model_args,
        data_args,
        training_args,
    ):
        '''
        This function prepares the dataset for every iteration.
        '''
        raw_datasets = selected_dataset

        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 512
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
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
    
        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = lm_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        return train_dataset


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

        ds = ds.map(tokenize, batched=False)
        ds = ds.filter(lambda x: len(x["input_ids"]) <= 256)

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
        alpha=0.2,
        iter_id=0,
        local_rank=0,
        output_min_length=16,
        output_max_length=48,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        :param batch_input: input prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()

        query_tensors = batch_input['input_ids']
        querys = batch_input['input']
        data_size = len(querys)

        reward_eva = []  # record the reward of the samples
        input_texts = []
        responses = []

        for i, query_tensor in enumerate(query_tensors):
            query = querys[i]
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
                texts_for_rewards = [q + r for q, r in zip(input_texts, generated_texts)]

                texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                    "type": "text_only",
                    "instances": [
                        { "text": text } for text in texts_for_rewards
                    ],
                })

                reward_dataset = reward_model.inference(texts_for_reward_dataset)
                rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]

                reward_eva.extend(rewards)
                responses.extend(generated_texts)
                input_texts = []

        data = [{"input": querys[j], "output": [responses[j]]} for j in range(len(reward_eva))]

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size


        data_to_send = [[data[i], reward_eva[i]] for i in range(len(data))]
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]]
            gathered_reward.extend(tmp_reward)

        idx = np.argsort(gathered_reward)[::-1][:int(len(gathered_reward) * alpha)]
        gathered_data = [gathered_data[j] for j in idx]
        reward_train = [gathered_reward[j] for j in idx]

        self.reward_seq.append(np.mean(gathered_reward))
        self.train_reawrd.append(np.mean(reward_train))
        import matplotlib.pyplot as plt
        if training_args.local_rank == 0:
            plt.plot(self.reward_seq, marker="o")
            plt.plot(self.train_reawrd, marker="*")
            plt.legend(["Model reward", "Reward of SFT Set"])
            plt.savefig(self.store_dir + '/training_reward.png')
            plt.close()

        logger.info(f"collected data of {len(gathered_data)}")
        logger.info([np.mean(gathered_reward), np.mean(reward_train)])

        if training_args.local_rank == 0 and output_reward_path is not None:
            with open(output_reward_path, mode='a') as fout:
                fout.write('mean reward: ' + str(np.mean(gathered_reward)) + 'mean reward in training set: ' + str(np.mean(reward_train)))
                fout.write("\n")


        prompt_structure = "{definition}{input}{output}"
        tmp_output_dataset = {
            "text": [ prompt_structure.format(
                          definition="", input=sample["input"], output=sample["output"][0]
                      ) for sample in gathered_data
            ]
        }

        # We store the training set for monitoring the RAFT training
        all_texts = tmp_output_dataset['text']
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = [{'text': i_text} for i_text in all_texts]
        import json
        if local_rank == 0:
            with open(self.store_dir + "/train_set_" + str(iter_id) + ".json", 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

        
        # We need to make sure that the order of the samples are the same for each agent
        all_process_list = [{}] * world_size
        data_to_send = [tmp_output_dataset, local_rank]
        dist.all_gather_object(all_process_list, data_to_send)
        for i in range(world_size):
            if all_process_list[i][1] == 0:
                output_dataset = all_process_list[i][0]
                break

        return DatasetDict({ "train": Dataset.from_dict(output_dataset) })

    def _get_batch_dataset_local(
            self,
            model,
            batch_input,
            K=8,
            iter_id=0,
            local_rank=0,
            output_min_length=16,
            output_max_length=48,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            reward_model=None,
            output_reward_path=None,
        ):
            """
            :param batch_input: input prompts
            """
            # we will get the batch dataset via Dataset.from_dict
            start_time = time.time()

            querys = batch_input['input']
            data_size = len(querys)

            reward_eva = []
            reward_train = []

            input_texts = []
            responses = []
            record_querys = []
            all_outputs = []

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
                texts_for_rewards = [q + r for q, r in zip(input_texts, generated_texts)]

                texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                    "type": "text_only",
                    "instances": [
                        { "text": texts_for_rewards[i] } for i in range(len(texts_for_rewards))
                    ],
                })

                reward_dataset = reward_model.inference(texts_for_reward_dataset)
                rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
                reward_eva.append(rewards[0])

                ################################
                # we impose some post-detection and discard the samples with certain criteria.
                for kk in range(K):
                    if self._discard_sample(generated_texts[kk]):
                        rewards[kk] = -self.INF
                ################################
                
                idx_to_record = np.argmax(rewards)
                all_outputs.append(generated_texts[0])

                # if we discard all the samples, we do not record the sample 
                if rewards[idx_to_record] != -self.INF:
                    responses.append(generated_texts[idx_to_record])
                    reward_train.append(rewards[idx_to_record])
                    record_querys.append(query)
                input_texts = []


            data = []
            for j in range(len(reward_train)):
                sample = {}
                sample["input"] = record_querys[j]
                sample["output"] = [responses[j]]
                data.append(sample)


            world_size = int(os.getenv("WORLD_SIZE", "1"))
            all_process_data =[{}] * world_size
            dist.all_gather_object(all_process_data, data)

            all_process_eval_reward =[{}] * world_size
            dist.all_gather_object(all_process_eval_reward, reward_eva)
            all_process_train_set_reward =[{}] * world_size
            dist.all_gather_object(all_process_train_set_reward, reward_train)

            
            gathered_data = []
            gathered_reward = []
            gathered_train_reward = []

            for i in range(world_size):
                gathered_data.extend(all_process_data[i])
                gathered_reward.extend(all_process_eval_reward[i])
                gathered_train_reward.extend(all_process_train_set_reward[i])

            if training_args.local_rank == 0 and output_reward_path is not None:
                with open(output_reward_path, mode='a') as fout:
                    fout.write('mean reward: ' + str(np.mean(gathered_reward)) + 'mean reward in training set: ' + str(np.mean(gathered_train_reward)))
                    fout.write("\n")
            logger.info([np.mean(gathered_reward), np.mean(gathered_train_reward)])

            
            self.reward_seq.append(np.mean(gathered_reward))
            self.train_reawrd.append(np.mean(reward_train))
            import matplotlib.pyplot as plt
            if training_args.local_rank == 0:
                plt.plot(self.reward_seq, marker="o")
                plt.plot(self.train_reawrd, marker="*")
                plt.legend(["Model reward", "Reward of SFT Set"])
                plt.savefig(self.store_dir + '/training_reward.png')
                plt.close()
            

            prompt_structure = "{definition}{input}{output}"
            tmp_output_dataset = {
                "text": [ prompt_structure.format(
                            definition="", input=sample["input"], output=sample["output"][0]
                        ) for sample in gathered_data
                ]
            }

            # We store the training set for monitoring the RAFT training
            all_texts = tmp_output_dataset['text']
            output_eval_dataset = {}
            output_eval_dataset['type'] = 'text_only'
            output_eval_dataset['instances'] = [{'text': i_text} for i_text in all_texts]
            import json
            if local_rank == 0:
                with open(self.store_dir + "/train_set_" + str(iter_id) + ".json", 'w', encoding='utf8') as f:
                    json.dump(output_eval_dataset, f, ensure_ascii=False)

            
            # We need to make sure that the order of the samples are the same for each agent
            all_process_list = [{}] * world_size
            data_to_send = [tmp_output_dataset, local_rank]
            dist.all_gather_object(all_process_list, data_to_send)
            for i in range(world_size):
                if all_process_list[i][1] == 0:
                    output_dataset = all_process_list[i][0]
                    break

            logger.info(f"collected data of {len(output_dataset['text'])}")


            return DatasetDict({ "train": Dataset.from_dict(output_dataset) })


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

        dataset = self._load_input_dataset(dataset, tokenizer)
        set_caching_enabled(False)

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


        set_seed(42 + training_args.local_rank)
        ITERATION = aligner_args.num_raft_iteration
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

        print(M, K)
        if training_args.local_rank == 0:
            print(aligner_args)
        self.store_dir = aligner_args.output_dir
        self.reward_seq = []
        self.train_reawrd = []
        
        data_size = len(dataset['input'])
        lr = training_args.learning_rate
        random_idxs = np.arange(data_size)
        np.random.shuffle(random_idxs)

        raft_trainer = self._initialize_trainer(model, tokenizer, training_args)
        raft_trainer.train(resume_from_checkpoint=False, is_first_time=True)

        for iteration in range(ITERATION):
            set_seed(666 + training_args.local_rank + world_size * (iteration+1))

            end_idx = np.min([data_size, (iteration+1) * M])
            batch_input = dataset.select(random_idxs[iteration * M : end_idx])
            model.gradient_checkpointing_disable()
            model.config.use_cache = True

            start_time = time.time()
            if collection_strategy == "top":
                selected_dataset = self._get_batch_dataset_top(
                    raft_trainer.tmp_model,
                    batch_input,
                    alpha,
                    iteration,
                    training_args.local_rank,
                    output_min_length=aligner_args.output_min_length,
                    output_max_length=aligner_args.output_max_length,
                    infer_batch_size=aligner_args.inference_batch_size_per_device,
                    generation_kwargs=generation_kwargs,
                    tokenizer=tokenizer,
                    training_args=training_args,
                    reward_model=reward_model,
                    output_reward_path=aligner_args.output_reward_path,
                )
            elif collection_strategy == "local":
                selected_dataset = self._get_batch_dataset_local(
                    raft_trainer.tmp_model,
                    batch_input,
                    K,
                    iteration,
                    training_args.local_rank,
                    output_min_length=aligner_args.output_min_length,
                    output_max_length=aligner_args.output_max_length,
                    infer_batch_size=K,
                    generation_kwargs=generation_kwargs,
                    tokenizer=tokenizer,
                    training_args=training_args,
                    reward_model=reward_model,
                    output_reward_path=aligner_args.output_reward_path,
                )
            end_time = time.time()
            logger.info("It takes %.2f s to inference one stage", end_time - start_time)
            
            raft_trainer.train_dataset = self._load_dataset(
                selected_dataset,
                raft_trainer.tmp_model,
                tokenizer,
                model_args,
                data_args,
                training_args,
            )

            logger.info(f"iter {iteration}")
            start_time = time.time()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False


            train_result = raft_trainer.train(resume_from_checkpoint=False)
            end_time = time.time()
            logger.info("It takes %.2f s to train one stage", end_time - start_time)
            if (iteration+1) * M  > data_size:
                logger.info("One epoch is completed.")
                break

            '''
            if training_args.local_rank == 0 and iteration % 2 == 0:
                wrapped_model.save(aligner_args.output_dir + "/" + "model" + str(iteration))
                print(iteration, "I save a model with", self.reward_seq[-1])
            '''

        if aligner_args.output_dir is not None:
            wrapped_model.save(aligner_args.output_dir)

        return wrapped_model 
