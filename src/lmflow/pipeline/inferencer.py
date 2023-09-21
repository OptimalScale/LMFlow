#!/usr/bin/env python
# coding=utf-8
"""The Inferencer class simplifies the process of model inferencing."""

import copy
import os
import torch
import wandb
import deepspeed
import sys
import numpy as np
import datetime
import json
import time
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoConfig
import torch.distributed as dist
import torch.nn.functional as F

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.utils.data_utils import (set_random_seed, batchlize,
                                     answer_extraction, process_image_flag)
from lmflow.utils.constants import IMAGE_TOKEN_INDEX
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
def rstrip_partial_utf8(string):
    return string.replace("\ufffd", "")

supported_dataset_type = [
    "text_only",
    "image_text",
]

logger = logging.getLogger(__name__)

class Inferencer(BasePipeline):
    """
    Initializes the `Inferencer` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    inferencer_args : InferencerArguments object.
        Contains the arguments required to perform inference.


    """
    def __init__(self, model_args, data_args, inferencer_args):
        self.data_args = data_args
        self.inferencer_args = inferencer_args
        self.model_args = model_args

        set_random_seed(self.inferencer_args.random_seed)

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if inferencer_args.device == "gpu":
            torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
            deepspeed.init_distributed()
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "15000"
            dist.init_process_group(
                "gloo", rank=self.local_rank, world_size=self.world_size
            )

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        try:
            self.model_hidden_size = self.config.hidden_size
        except:
            print("Error in setting hidden size, use the default size 1024")
            self.model_hidden_size = 1024 # gpt2 seems do not have hidden_size in config


    def create_dataloader(self, dataset: Dataset):
        r"""Batchlize dataset and format it to dataloader.

        Args:
            dataset (Dataset): the dataset object

        Output:
            dataloader (batchlize): the dataloader object
            dataset_size (int): the length of the dataset

        """
        if dataset.get_type() == "text_only":
            data_dict = dataset.to_dict()
            inputs = [instance["text"] for instance in data_dict["instances"] ]
        elif dataset.get_type() == "image_text":
            inputs = dataset.to_list()

        dataset_size = len(inputs)
        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({
                "input": inputs[idx],
                "input_idx": idx
            })

        dataloader = batchlize(
            dataset_buf,
            batch_size=1,
            random_shuffle=False,
        )
        return dataloader, dataset_size


    def inference(
        self,
        model,
        dataset: Dataset,
        max_new_tokens: int=100,
        temperature: float=0.0,
        prompt_structure: str='{input}',
        remove_image_flag: bool=False,
        chatbot_type: str="mini_gpt",
    ):
        """
        Perform inference for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform inference

        dataset : Dataset object.


        Returns:

        output_dataset: Dataset object.
        """
        if dataset.get_type() not in supported_dataset_type:
            raise NotImplementedError(
                'input dataset should have type {}'.format(
                                        supported_dataset_type))
        dataloader, data_size = self.create_dataloader(dataset)

        # The output dataset
        output_dict = {
            "type": "text_only",
            "instances": [
            ]
        }

        for batch_index, batch in enumerate(dataloader):
            current_batch = batch[0]        # batch size is 1
            if isinstance(current_batch['input'], str):
                input = prompt_structure.format(input=current_batch['input'])
            else:
                input = current_batch['input']
                input['text'] = prompt_structure.format(input=input['text'])

            if 'images' in input and isinstance(input['images'], list):
                input['images'] = np.array(input['images'])
            if remove_image_flag:
                # remove the image flag <ImageHere> in tokenization;
                if chatbot_type == "mini_gpt":
                    image_split_flag = "<ImageHere>"
                elif chatbot_type:
                    image_split_flag = "<image>"
                else:
                    raise NotImplementedError
                input['text'] = input['text'].split(image_split_flag)
                # TODO remove this code by update the tokenizer
                input_ids = []
                attention_mask = []
                image_token_indexes = []
                temp_input = copy.deepcopy(input)
                for idx in range(len(input['text'])):
                    temp_input['text'] = input['text'][idx]
                    temp_inputs = model.encode(
                        temp_input,
                        return_tensors="pt",
                        add_special_tokens=idx == 0
                    ).to(device=self.local_rank)
                    input_ids.append(temp_inputs['input_ids'])
                    attention_mask.append(temp_inputs['attention_mask'])
                    if chatbot_type == "llava":
                        # add the flag for inserting the image.
                        # TODO should merge the way of handling image flag in minigpt and llava.
                        index_tensor = torch.tensor(
                            [IMAGE_TOKEN_INDEX]
                        ).to(device=self.local_rank)
                        index_tensor = index_tensor.reshape(1, 1)
                        input_ids.append(index_tensor)
                        attention_mask.append(
                            torch.ones(1,1).to(device=self.local_rank))
                    image_token_indexes.append(
                        temp_inputs["input_ids"].shape[1])
                if len(image_token_indexes) > 1:
                    image_token_indexes = image_token_indexes[:-1]
                    if chatbot_type == "llava":
                        input_ids = input_ids[:-1]
                        attention_mask = attention_mask[:-1]
                inputs = temp_inputs
                inputs["input_ids"] = torch.cat(input_ids, dim=1)
                inputs["attention_mask"] = torch.cat(attention_mask, dim=1)
            else:
                if self.inferencer_args.device == "gpu":
                    inputs = model.encode(
                        input, return_tensors="pt"
                    ).to(device=self.local_rank)
                elif self.inferencer_args.device == "cpu":
                    inputs = model.encode(
                        input, return_tensors="pt"
                    ).to(device='cpu')
                else:
                    raise NotImplementedError(
                        f"device \"{self.inferencer_args.device}\" is not supported"
                    )
            if remove_image_flag:
                inputs["image_token_indexes"] = image_token_indexes
                inputs["one_sample_multiple_images"] = True

            outputs = model.inference(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.inferencer_args.temperature,
                repetition_penalty=self.inferencer_args.repetition_penalty,
                do_sample=self.inferencer_args.do_sample,
            )

            # only return the generation, trucating the input
            if self.model_args.arch_type != "vision_encoder_decoder":
                text_out = model.decode(outputs[0], skip_special_tokens=True)
                prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
                text_out = text_out[prompt_length:]
            else:
                # to avoid redundant/missing leading space problem, we use a
                # part of the input text
                input_text = inputs['input_ids'][0][-1:]
                text_out = model.decode(torch.cat([input_text, outputs[0]]), skip_special_tokens=True)
                prompt_length = len(model.decode(input_text, skip_special_tokens=True,))
                text_out = text_out[prompt_length:]

            output_dict["instances"].append({ "text": text_out })

        output_dataset = Dataset(DatasetArguments(dataset_path = None))
        output_dataset = output_dataset.from_dict(output_dict)

        return output_dataset

    def stream_inference(
        self,
        context,
        model,
        max_new_tokens,
        token_per_step,
        temperature,
        end_string,
        input_dataset,
        remove_image_flag: bool=False,
    ):
        response = ""
        history = []
        if "ChatGLMModel" in self.config.architectures:
            for response, history in model.get_backend_model().stream_chat(model.get_tokenizer(), context, history=history):
                response = rstrip_partial_utf8(response)
                yield response, False
        else:
            for _ in range(0, self.inferencer_args.max_new_tokens // token_per_step):
                output_dataset = self.inference(
                    model=model,
                    dataset=input_dataset,
                    max_new_tokens=token_per_step,
                    temperature=self.inferencer_args.temperature,
                    remove_image_flag=remove_image_flag,
                )

                new_append_text = output_dataset.to_dict()["instances"][0]["text"]
                new_append_text = rstrip_partial_utf8(new_append_text)
                response += new_append_text

                input_dict = input_dataset.to_dict()
                input_dict["instances"][0]["text"] += new_append_text
                input_dataset = input_dataset.from_dict(input_dict)

                flag_break = False
                try:
                    index = response.index(end_string)
                    flag_break = True
                except ValueError:
                    response += end_string
                    index = response.index(end_string)

                response = response[:index]

                yield response, flag_break


class SpeculativeInferencer(Inferencer):
    """
    Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192)

    Parameters
    ------------
    target_model_args : ModelArguments object.
        Contains the arguments required to load the target model.
        
    draft_model_args : ModelArguments object.
        Contains the arguments required to load the draft model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    inferencer_args : InferencerArguments object.
        Contains the arguments required to perform inference.


    """
    def __init__(self, model_args, draft_model_args, data_args, inferencer_args):
        super().__init__(model_args, data_args, inferencer_args)
        self.draft_model_args = draft_model_args

        self.draft_config = AutoConfig.from_pretrained(draft_model_args.model_name_or_path, trust_remote_code=True)
        try: 
            self.draft_model_hidden_size = self.draft_config.hidden_size
        except:
            print("Error in setting hidden size for draft model, use the default size 1024")
            self.draft_model_hidden_size = 768
            
    
    @staticmethod        
    def score_to_prob(scores: torch.Tensor, 
                      temperature: float = 0.,
                      top_p: float = 1.,) -> torch.Tensor:
        """Convert scores (NOT softmaxed tensor) to probabilities with support for temperature, top-p sampling, and argmax.

        Parameters
        ----------
        scores : torch.Tensor
            Input scores.
        temperature : float, optional
            Temperature parameter for controlling randomness. Higher values make the distribution more uniform, 
            lower values make it peakier. When temperature <= 1e-6, argmax is used. by default 0.0
        top_p : float, optional
            Top-p sampling parameter for controlling the cumulative probability threshold, by default 1.0 (no threshold)

        Returns
        -------
        torch.Tensor
            Probability distribution after adjustments.
        """
        assert temperature >= 0.0
        assert 0.0 < top_p <= 1.0
        
        if temperature <= 1e-6:
            final_prob = F.one_hot(scores.argmax(dim=1), num_classes=scores.size(1)).float()
        else:
            scores /= temperature
            if top_p < 1.0:
                sorted_scores, _ = torch.sort(scores, descending=True)
                probs = sorted_scores.softmax(dim=1)
                cumulative_probs = torch.cumsum(probs, dim=1)
                mask = cumulative_probs <= top_p
                if mask.any():
                    thresholded_probs = probs * mask
                    thresholded_probs = thresholded_probs / thresholded_probs.sum(dim=1, keepdim=True)
                    final_prob = torch.zeros_like(scores)
                    final_prob.scatter_add_(1, sorted_scores.argsort(dim=1), thresholded_probs)
                else:
                    final_prob = scores.softmax(dim=1)
                    
            else:
                final_prob = scores.softmax(dim=1)

        return final_prob
    
    
    @staticmethod
    def sample(prob: torch.Tensor, num_samples: int = 1) -> Dict:
        """Sample from a tensor of probabilities
        """
        sampled_indices = torch.multinomial(prob, num_samples=num_samples, replacement=True) 
        return {'sampled_token': sampled_indices, 'sampled_prob': prob.gather(dim=1, index=sampled_indices), 'all_prob': prob}
    
    
    @staticmethod
    def predict_next_token(model: HFDecoderModel, input_ids: torch.Tensor, num_new_tokens: int = 1):
        """Predict the next token given the input_ids.
        """
        output = model.inference(input_ids, 
                                 use_accelerator=True, 
                                 max_new_tokens=num_new_tokens,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 do_sample=True,
                                 num_beams=1)
        return output
    
    
    def autoregressive_sampling(self, 
                                input_ids: torch.Tensor, 
                                model: HFDecoderModel, 
                                temperature: float = 0., 
                                num_new_tokens: int = 5) -> Dict:
        """Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192) Section 2.2
        """
        sequence = input_ids
        new_tokens = []
        
        for _ in range(num_new_tokens):
            pred = self.predict_next_token(model=model, input_ids=sequence, num_new_tokens=1) # predict next one token
            prob = self.score_to_prob(pred.scores[0], temperature=temperature)
            sampled = self.sample(prob=prob, num_samples=1)
            new_tokens.append(sampled)
            sequence = torch.cat([sequence, sampled['sampled_token']], dim=1)
            
        return {"sequence": sequence, "new_tokens": new_tokens}
    
    
    def inference(
        self,
        model: HFDecoderModel,
        draft_model: HFDecoderModel,
        input: str,
        temperature: float = 0.,
        gamma: int = 5,
        max_new_tokens: int = 100,
    ):
        """
        Perform inference for a model

        Parameters
        ------------
        model : HFDecoderModel object.
            TunableModel to verify tokens generated by the draft model.
            
        draft_model : HFDecoderModel object.
            TunableModel that provides approximations of the target model.

        input : str.
            The input text (i.e., the prompt) for the model.
            
        gamma : int.
            The number of tokens to be generated by the draft model within each iter.
            
        max_new_tokens : int.
            The maximum number of tokens to be generated by the target model.
            

        Returns
        -------
        output: str.
            The output text generated by the model.
        """
        assert gamma > 0

        if self.inferencer_args.device == "gpu":
            inputs = model.encode(input, return_tensors="pt").to(device=self.local_rank)
        elif self.inferencer_args.device == "cpu":
            inputs = model.encode(input, return_tensors="pt").to(device='cpu')
        else:
            raise NotImplementedError(
                f"device \"{self.inferencer_args.device}\" is not supported"
            )


        def speculative_sampling(input_ids: torch.Tensor,
                                 model: HFDecoderModel,
                                 draft_model: HFDecoderModel,
                                 temperature: float = 0.) -> torch.Tensor:
            """Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192)

            Parameters
            ----------
            input_ids : torch.Tensor
            draft_model : TunableModel object
            model_list : List[TunableModel object]

            Returns
            -------
            torch.Tensor
            """
            len_input_ids= input_ids.shape[1]
            logger.debug(f"len of input_ids: {len_input_ids}")
            
            # STEP 1: Sample γ guesses x1, ..., xγ from Mq (draft model) autoregressively
            output_draft = self.autoregressive_sampling(input_ids=input_ids, model=draft_model, num_new_tokens=gamma)
            logger.debug(f"draft result: {output_draft['sequence']}")
            logger.debug(f"draft result decoded: {draft_model.decode(output_draft['sequence'][0])}")
            
            
            # STEP 2: Run Mp (target model) in parallel
            # generate sequences [prefix, x1, x2, ..., xγ]
            output = model.get_backend_model()(input_ids=output_draft['sequence'], return_dict=True)
            logger.debug(f'shape of output: {output.logits.shape}')
            
            
            # STEP 3: Determine the number of accepted guesses n
            accepted = [False] * gamma
            for i in range(gamma):
                draft_sampled_token_id = output_draft['new_tokens'][i]['sampled_token']
                draft_sampled_token_prob = output_draft['new_tokens'][i]['sampled_prob']
                token_prob = self.score_to_prob(output.logits[:,len_input_ids+i-1,:], temperature=temperature)[0, draft_sampled_token_id]

                # reject the sample with probability 1 - p(x)/q(x)
                if torch.rand_like(token_prob) > token_prob/draft_sampled_token_prob:
                    break
                else:
                    accepted[i] = True
                
            logger.debug(f"Speculative Sampling: Accepted: {sum(accepted)}/{gamma}")


            # STEP 4: Adjust the distribution from Mp if needed
            if not all(accepted):
                all_prob = self.score_to_prob(output.logits[:,len_input_ids+i-1,:], temperature=temperature)
                draft_all_prob = output_draft['new_tokens'][i]['all_prob']
                adjusted_prob = torch.max(torch.zeros_like(all_prob), all_prob - draft_all_prob)
                prob = adjusted_prob / adjusted_prob.sum(dim=1, keepdim=True)
            else:
                prob = self.score_to_prob(output.logits[:,-1,:], temperature=temperature)


            # STEP 5: Return n tokens from Mq, and one token from Mp
            token_from_target_model = self.sample(prob)['sampled_token']
            final_sequence = torch.concat([output_draft['sequence'][:,:len_input_ids+sum(accepted)], token_from_target_model], dim=1)

            return final_sequence
        

        num_generated_new_tokens = 0
        len_raw_input = len(inputs[0])
        while num_generated_new_tokens < max_new_tokens:
            logger.debug(f'===== New iter =====')
            logger.debug(f"input_ids: {inputs}")
            sampling_result = speculative_sampling(input_ids=inputs,
                                                   model=model,
                                                   draft_model=draft_model,
                                                   temperature=temperature)
            logger.debug(f'sampling result: {sampling_result}')
            logger.debug(f'sampling result decoded: {model.decode(sampling_result[0])}')
            num_generated_new_tokens += len(sampling_result[0]) - len(inputs[0])
            inputs = sampling_result
        
        
        # if, say, num_generated_new_tokens = 19, and the model accept 3 
        # tokens, the actual generated tokens would be 22.
        return model.decode(inputs[0,:len_raw_input+max_new_tokens])
        

    def stream_inference(self):
        raise NotImplementedError("Streaming output for SpeculativeInferencer is not supported yet")
