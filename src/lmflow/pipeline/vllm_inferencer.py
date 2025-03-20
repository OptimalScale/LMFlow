#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import copy
import importlib.resources as pkg_resources
import json
import logging
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import subprocess
import sys
from functools import partial
from typing import List, Union, Optional, Dict, Any

import numpy as np
from transformers import AutoTokenizer

from lmflow.datasets import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.args import (
    InferencerArguments, 
    ModelArguments, 
    DatasetArguments,
)
from lmflow.utils.common import make_shell_args_from_dataclass
from lmflow.utils.constants import RETURN_CODE_ERROR_BUFFER, MEMORY_SAFE_VLLM_INFERENCE_ENV_VAR_TO_REMOVE
from lmflow.utils.data_utils import VLLMInferenceResultWithInput
from lmflow.utils.versioning import is_vllm_available, is_ray_available


logger = logging.getLogger(__name__)


if is_vllm_available():
    from vllm import SamplingParams, LLM
else:
    raise ImportError("VLLM is not available, please install vllm.")

if is_ray_available():
    import ray
    import ray.data
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    logger.warning("Ray is not available, distributed vllm inference will not be supported.")
    

class InferencerWithOffloading(BasePipeline):
    def __init__(
        self, 
        model_args: ModelArguments,
        data_args: DatasetArguments,
        inferencer_args: InferencerArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
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
        data_args: DatasetArguments,
        inferencer_args: InferencerArguments,
    ):
        assert inferencer_args.use_vllm, "The inferencer_args.use_vllm must be True."
        super().__init__(model_args, data_args, inferencer_args)
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
        enable_decode_inference_result: bool = True,
        release_gpu: bool = False,
        inference_args: Optional[InferencerArguments] = None,
        enable_distributed_inference: bool = False,
        **kwargs,
    ) -> List[VLLMInferenceResultWithInput]:
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
        enable_decode_inference_result : bool, optional
            Whether to decode after generation, by default False.
        release_gpu : bool, optional
            Whether to release gpu resources, by default False. 
        inference_args : InferencerArguments, optional
            by default None

        Returns
        -------
        List[VLLMInferenceResultWithInput]
            Return a list of VLLMInferenceResultWithInput, where each
            element contains the input prompt and the corresponding output.
            
            When `enable_decode_inference_result = True`, the output would be a list of strings,
            contains sampling_params.n samples for the corresponding prompt.
            
            When `enable_decode_inference_result = False`, return a list of list of ints 
            (token ids, no decoding after generation).
        """
        if inference_args:
            logger.warning(
                "Overriding the default inference arguments with the provided arguments in .inference()"
            )
            sampling_params = self.parse_to_sampling_params(inference_args)
        else:
            sampling_params = self.sampling_params
            
        sampling_params.detokenize = enable_decode_inference_result
        
        model_input = model.prepare_inputs_for_inference(
            dataset=dataset,
            apply_chat_template=self.inferencer_args.apply_chat_template,
            use_vllm=self.inferencer_args.use_vllm,
            enable_distributed_inference=enable_distributed_inference,
        )
        
        if enable_distributed_inference:
            outputs = self._distributed_inference(
                model=model, 
                model_input=model_input, 
                sampling_params=sampling_params,
                num_instances=kwargs.get("distributed_inference_num_instances"),
                batch_size=kwargs.get("inference_batch_size", 4),
                release_gpu=release_gpu,
            )
        else:
            outputs = self._inference(
                model=model,
                model_input=model_input, 
                sampling_params=sampling_params,
                release_gpu=release_gpu,
            )

        if self.inferencer_args.save_results:
            self.save_inference_results(outputs, self.inferencer_args.results_path)
            
        return outputs


    def _inference(
        self,
        model: HFDecoderModel, 
        model_input: List[str], 
        sampling_params: SamplingParams,
        release_gpu: bool = False,
    ) -> List[VLLMInferenceResultWithInput]:        
        outputs = model.inference(
            inputs=model_input,
            sampling_params=sampling_params,
            release_gpu=release_gpu,
            use_vllm=True,
            vllm_gpu_memory_utilization=self.inferencer_args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=self.inferencer_args.vllm_tensor_parallel_size,
        )
        
        return outputs
    
    
    def _distributed_inference(
        self,
        model: HFDecoderModel, 
        model_input: ray.data.Dataset, 
        sampling_params: SamplingParams,
        num_instances: int,
        batch_size: int = 4,
        release_gpu: bool = False,
    ) -> List[VLLMInferenceResultWithInput]:
        # prepare distributed inference resources
        # from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_distributed.py
        ## strategy
        def scheduling_strategy_fn():
            # One bundle per tensor parallel worker
            pg = ray.util.placement_group(
                [{
                    "GPU": 1,
                    "CPU": 1
                }] * self.inferencer_args.vllm_tensor_parallel_size,
                strategy="STRICT_PACK",
            )
            return dict(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    pg, placement_group_capture_child_tasks=True
                )
            )
            
        resources_kwarg: Dict[str, Any] = {}
        if self.inferencer_args.vllm_tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            resources_kwarg["num_gpus"] = 1
        else:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            resources_kwarg["num_gpus"] = 0
            resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn
            
        ## predictor
        class DistributedPredictor:
            def __init__(
                self, 
                model: HFDecoderModel,
                sampling_params: SamplingParams,
                vllm_gpu_memory_utilization: float,
                vllm_tensor_parallel_size: int,
                release_gpu: bool=False,
            ):
                self.model = copy.deepcopy(model)
                self.model.activate_model_for_inference(
                    use_vllm=True,
                    vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                    vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                )
                self.sampling_params = sampling_params
                self.release_gpu = release_gpu
                
            def __call__(self, batch: Dict[str, np.ndarray]):
                """batch: Dict[str, np.ndarray], {"item": array(['...', '...', '...', ...])}
                """
                batched_inference_res = self.model.inference(
                    inputs=batch['item'],
                    sampling_params=self.sampling_params,
                    release_gpu=self.release_gpu,
                    use_vllm=True,
                ) # this is the postprocessed output, see model.__vllm_inference
                batched_final_res = {
                    "input": [sample['input'] for sample in batched_inference_res],
                    "output": [sample['output'] for sample in batched_inference_res] 
                } # do this since we're writing to a pandas dataframe
                return batched_final_res
            
        # inference        
        model_input_mapping = model_input.map_batches(
            DistributedPredictor,
            concurrency=num_instances, # Set the concurrency to the number of LLM instances.
            batch_size=batch_size,
            fn_constructor_kwargs={
                "model": model,
                "sampling_params": sampling_params,
                "vllm_gpu_memory_utilization": self.inferencer_args.vllm_gpu_memory_utilization,
                "vllm_tensor_parallel_size": self.inferencer_args.vllm_tensor_parallel_size,
                "release_gpu": release_gpu,
            },
            **resources_kwarg,
        )
        
        df_model_output = model_input_mapping.to_pandas() # the actual forwards are executed here
        logger.info(f"Distributed vllm inference result preview:\n{df_model_output.head(10)}")
        
        model_output = [
            {"input": row["input"], "output": row["output"]} for _, row in df_model_output[:].iterrows()
        ]
        
        return model_output
    
    
    def save_inference_results(
        self,
        outputs: Union[List[List[str]], List[List[List[int]]]],
        save_file_path: str,
    ):
        with open(save_file_path, "w", encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)
            
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
        super().__init__(model_args, data_args, inferencer_args)
        self.inferencer_file_path = pkg_resources.files("lmflow.pipeline.utils") / "memory_safe_vllm_inference.py"
        
    
    def inference(self) -> List[VLLMInferenceResultWithInput]:
        inferencer_args = make_shell_args_from_dataclass(
            dataclass_objects=[
                self.model_args, 
                self.data_args, 
                self.inferencer_args,
            ],
            format="shell",
        )
        cmd = "python " + str(self.inferencer_file_path) + " " + inferencer_args
        current_env = os.environ.copy()
        for var in MEMORY_SAFE_VLLM_INFERENCE_ENV_VAR_TO_REMOVE:
            current_env.pop(var, None)
        
        cli_res = subprocess.run(
            args=cmd,
            stdout=sys.stdout,
            stderr=sys.stdout,
            shell=True,
            preexec_fn=os.setsid,
            env=current_env,
        )
        logger.info(f"MemorySafeVLLMInference subprocess run finished, info at finish: {cli_res}")
        
        if cli_res.returncode in RETURN_CODE_ERROR_BUFFER:
            # > Fatal Python error: _enter_buffered_busy: could not acquire lock for <_io.BufferedWriter name='<stdout>'> 
            # > at interpreter shutdown, possibly due to daemon threads
            logger.warning(
                "^^^^^^^^^^ Please ignore the above error, as it comes from the subprocess. "
                "This may due to a kill signal with unfinished stdout/stderr writing in the subprocess. "
            )
        else:
            if cli_res.returncode != 0:
                raise RuntimeError(f"Error during MemorySafeVLLMInference: {cli_res}")
                
        outputs = self.load_inference_results(self.inferencer_args.results_path)
        logger.info("MemorySafeVLLMInference result captured.")
        
        return outputs