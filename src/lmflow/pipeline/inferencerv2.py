#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import os
import sys
import signal
import json
from pathlib import Path
import logging
import subprocess
import importlib.resources as pkg_resources
from typing import List, Union, Optional

from vllm import SamplingParams
from transformers import AutoTokenizer

from lmflow.datasets import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.args import InferencerArguments, ModelArguments, DatasetArguments
from lmflow.pipeline.utils.collections import make_shell_args_from_dataclass
from lmflow.utils.constants import MEMORY_SAFE_VLLM_INFERENCE_FINISH_FLAG


logger = logging.getLogger(__name__)


class InferencerWithOffloading(BasePipeline):
    def __init__(
        self, 
        model_args: ModelArguments,
        inferencer_args: InferencerArguments,
    ):
        self.model_args = model_args
        self.inferencer_args = inferencer_args
        self.eos_token_id = AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token_id

    def inference(self):
        raise NotImplementedError(".inference is not implemented")
        
    def save_inference_results(self):
        raise NotImplementedError(".save_inference_results is not implemented")
        
    def load_inference_results(self):
        raise NotImplementedError(".load_inference_results is not implemented")
    
    
class VLLMInferencer(InferencerWithOffloading):
    def __init__(
        self,
        model_args: ModelArguments,
        inferencer_args: InferencerArguments,
    ):
        assert model_args.use_vllm_inference, "The model_args.use_vllm_inference must be True."
        super().__init__(model_args, inferencer_args)
        self.sampling_params = self.parse_to_sampling_params(inferencer_args)
        
    
    def parse_to_sampling_params(
        self,
        inference_args: InferencerArguments,
    ) -> SamplingParams:
        return SamplingParams(
            use_beam_search=inference_args.use_beam_search,
            n=inference_args.num_output_sequences,
            temperature=inference_args.temperature + 1e-6,
            max_tokens=inference_args.max_new_tokens,
            seed=inference_args.random_seed,
            top_p=inference_args.top_p,
            top_k=inference_args.top_k,
            stop_token_ids=[self.eos_token_id] + inference_args.additional_stop_token_ids
        )


    def inference(
        self,
        model: HFDecoderModel, 
        dataset: Dataset, 
        detokenize: bool = True,
        release_gpu: bool = False,
        inference_args: Optional[InferencerArguments] = None,
    ) -> Union[List[List[str]], List[List[List[int]]]]:
        """Perform inference using the provided model and dataset. Will save inference results if
        `save_results` is set to True in `inferencer_args`.

        Parameters
        ----------
        model : HFDecoderModel
            LMFlow HFDecoderModel object
        dataset : Dataset
            LMFlow Dataset object
        apply_chat_template : bool, optional
            Whether to apply chat template to the input, by default True.
        detokenize : bool, optional
            Whether to decode after generation, by default False.
        release_gpu : bool, optional
            Whether to release gpu resources, by default False. 
            NOTE: The reason why `release_gpu` and `detokenize` are not in `inference_args` is that
            Inferencer may be used by other pipeline, and the pipeline may want to control these 
            two behaviors dynamically. 
        inference_args : InferencerArguments, optional
            by default None

        Returns
        -------
        Union[List[List[str]], List[List[List[int]]]]
            When `detokenize = True`, return a list of list of strings. Inner list
            contains inference_args.num_output_sequences samples for a single prompt 
            (i.e., `len(res[i]) = inference_args.num_output_sequences`). Outer list 
            contains the results for all prompts (i.e., `len(res) = len(dataset)`).
            
            When `detokenize = False`, return a list of list of list of ints 
            (token ids, no decoding after generation).
        """
        if inference_args:
            logger.warning(
                "Overriding the default inference arguments with the provided arguments in .inference()"
            )
            sampling_params = self.parse_to_sampling_params(inference_args)
        else:
            sampling_params = self.sampling_params
            
        sampling_params.detokenize = detokenize
        
        model_input = model.prepare_inputs_for_inference(
            dataset=dataset, 
            apply_chat_template=self.inferencer_args.apply_chat_template
        )
        
        outputs = model.inference(
            inputs=model_input,
            sampling_params=sampling_params,
            release_gpu=release_gpu,
        )
                
        if self.inferencer_args.save_results:
            self.save_inference_results(outputs, self.inferencer_args.results_path)
                    
        return outputs
    
    
    def save_inference_results(
        self,
        outputs: Union[List[List[str]], List[List[List[int]]]],
        save_file_path: str,
    ):
        with open(save_file_path, "w") as f:
            json.dump(outputs, f)
            
        logger.info(f"Inference results are saved to {save_file_path}.")
        
        
    def load_inference_results(
        self,
        results_path: str,
    ) -> Union[List[List[str]], List[List[List[int]]]]:
        with open(results_path, "r") as f:
            results = json.load(f)
            
        return results
        
        
class MemorySafeVLLMInferencer(VLLMInferencer):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        inferencer_args: InferencerArguments,
    ):
        assert inferencer_args.save_results, "For MemorySafeVLLMInferencer, `save_results` must be True."
        super().__init__(model_args, inferencer_args)
        self.data_args = data_args
        self.inferencer_file_path = pkg_resources.files("lmflow.pipeline.utils") / "memory_safe_vllm_inference.py"
        
    
    def inference(self):
        cuda_env = ''
        if self.inferencer_args.memory_safe_vllm_inference_devices:
            if len(self.inferencer_args.memory_safe_vllm_inference_devices.split(',')) != self.model_args.vllm_tensor_parallel_size:
                raise ValueError(
                    "The number of devices in `memory_safe_vllm_inference_devices` "
                    "must be equal to the `vllm_tensor_parallel_size`. "
                    f"Got {self.inferencer_args.memory_safe_vllm_inference_devices=} "
                    f"and {self.model_args.vllm_tensor_parallel_size=}."
                )
            cuda_env = f"CUDA_VISIBLE_DEVICES={self.inferencer_args.memory_safe_vllm_inference_devices}"
        inferencer_args = make_shell_args_from_dataclass(
            dataclass_objects=[
                self.model_args, 
                self.data_args, 
                self.inferencer_args,
            ],
            format="shell",
        )
        cmd = cuda_env + " python " + str(self.inferencer_file_path) + " " + inferencer_args
        
        cli_res = subprocess.run(
            args=cmd,
            stdout=sys.stdout,
            stderr=sys.stdout,
            shell=True,
            preexec_fn=os.setsid
        )
        logger.info(f"MemorySafeVLLMInference subprocess run finished, info at finish: {cli_res}")
        
        if cli_res.returncode != 0:
            raise RuntimeError(f"Error during MemorySafeVLLMInference.")
        else:
            outputs = self.load_inference_results(self.inferencer_args.results_path)
            logger.info("MemorySafeVLLMInference result captured.")
            
            return outputs