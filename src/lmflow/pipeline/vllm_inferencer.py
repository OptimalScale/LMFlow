#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import importlib.resources as pkg_resources
import logging
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import subprocess
import sys
from typing import Optional

from transformers import AutoTokenizer

from lmflow.args import (
    DatasetArguments,
    InferencerArguments,
    ModelArguments,
)
from lmflow.datasets import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.utils.common import make_shell_args_from_dataclass
from lmflow.utils.constants import MEMORY_SAFE_VLLM_INFERENCE_ENV_VAR_TO_REMOVE, RETURN_CODE_ERROR_BUFFER
from lmflow.utils.protocol import DataProto
from lmflow.utils.versioning import is_vllm_available

logger = logging.getLogger(__name__)


if is_vllm_available():
    pass
else:
    raise ImportError("VLLM is not available, please install vllm.")


class VLLMInferencer(BasePipeline):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        inferencer_args: InferencerArguments,
    ):
        assert inferencer_args.inference_engine == "vllm"
        self.model_args = model_args
        self.data_args = data_args
        self.inferencer_args = inferencer_args
        self.eos_token_id = AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token_id
        self.sampling_params = self._parse_args_to_sampling_params(inferencer_args)

    def _parse_args_to_sampling_params(
        self,
        inference_args: InferencerArguments,
    ) -> dict:
        if inference_args.use_beam_search:
            logger.warning("`use_beam_search` is ignored, as vLLM V1 engine no longer supports beam search.")

        sampling_params = {
            "n": inference_args.num_output_sequences,
            "temperature": inference_args.temperature + 1e-6,
            "max_new_tokens": inference_args.max_new_tokens,
            "seed": inference_args.random_seed,
            "top_p": inference_args.top_p,
            "top_k": inference_args.top_k,
            "stop_token_ids": [self.eos_token_id] + inference_args.additional_stop_token_ids,
        }

        return sampling_params

    def inference(
        self,
        model: HFDecoderModel,
        dataset: Dataset,
        release_gpu: bool = False,
        inference_args: Optional[InferencerArguments] = None,
    ) -> DataProto:
        if inference_args:
            logger.warning("Overriding the default inference arguments with the provided arguments in .inference()")
            sampling_params = self._parse_args_to_sampling_params(inference_args)
        else:
            sampling_params = self.sampling_params

        model_input = model.prepare_inputs_for_inference(
            dataset=dataset,
            apply_chat_template=self.inferencer_args.apply_chat_template,
            inference_engine="vllm",
            sampling_params=sampling_params,
        )

        outputs = model.inference(
            inputs=model_input,
            release_gpu=release_gpu,
            inference_engine="vllm",
            gpu_memory_utilization=self.inferencer_args.inference_gpu_memory_utilization,
            tensor_parallel_size=self.inferencer_args.inference_tensor_parallel_size,
            data_parallel_size=self.inferencer_args.inference_data_parallel_size,
            max_model_len=self.inferencer_args.inference_max_model_len,
        )

        if self.inferencer_args.save_inference_results:
            self.save_inference_results(outputs, self.inferencer_args.inference_results_path)

        return outputs

    def save_inference_results(
        self,
        outputs: DataProto,
        inference_results_path: str,
    ):
        save_path = os.path.join(inference_results_path, "inference_results.pkl")
        outputs.save_to_disk(save_path)
        logger.info(f"Inference results are saved to {save_path}.")

    def load_inference_results(
        self,
        inference_results_path: str,
    ) -> DataProto:
        load_path = os.path.join(inference_results_path, "inference_results.pkl")
        return DataProto.load_from_disk(load_path)


class MemorySafeVLLMInferencer(VLLMInferencer):
    """Run VLLM inference in a subprocess for memory safety.

    This is a workaround since vllm cannot release GPU memory properly
    in-process. See: https://github.com/vllm-project/vllm/issues/1908
    """

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        inferencer_args: InferencerArguments,
    ):
        assert inferencer_args.save_inference_results or inferencer_args.save_results, (
            "For MemorySafeVLLMInferencer, `save_inference_results` must be True."
        )
        super().__init__(model_args, data_args, inferencer_args)
        self.inferencer_file_path = pkg_resources.files("lmflow.pipeline.utils") / "memory_safe_vllm_inference.py"

    def inference(self) -> DataProto:
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
            logger.warning(
                "^^^^^^^^^^ Please ignore the above error, as it comes from the subprocess. "
                "This may due to a kill signal with unfinished stdout/stderr writing in the subprocess. "
            )
        else:
            if cli_res.returncode != 0:
                raise RuntimeError(f"Error during MemorySafeVLLMInference: {cli_res}")

        inference_results_path = self.inferencer_args.inference_results_path or self.inferencer_args.results_path
        outputs = self.load_inference_results(inference_results_path)
        logger.info("MemorySafeVLLMInference result captured.")

        return outputs
