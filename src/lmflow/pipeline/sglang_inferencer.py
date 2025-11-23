#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import json
import logging

from typing import Optional, Union

from transformers import AutoTokenizer

from lmflow.args import (
    DatasetArguments,
    InferencerArguments,
    ModelArguments,
)
from lmflow.datasets import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.utils.versioning import is_sglang_available

logger = logging.getLogger(__name__)


if is_sglang_available():
    pass
else:
    raise ImportError("SGLang is not available, please install sglang using `pip install -e .[sglang]`.")


class SGLangInferencer(BasePipeline):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        inferencer_args: InferencerArguments,
    ):
        assert inferencer_args.inference_engine == "sglang"
        self.model_args = model_args
        self.data_args = data_args
        self.inferencer_args = inferencer_args
        self.eos_token_id = AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token_id
        self.sampling_params = self._parse_args_to_sampling_params(inferencer_args)

    def _parse_args_to_sampling_params(
        self,
        inference_args: InferencerArguments,
    ) -> dict:
        sampling_params = {
            "use_beam_search": inference_args.use_beam_search,
            "n": inference_args.num_output_sequences,
            "temperature": inference_args.temperature + 1e-6,
            "max_tokens": inference_args.max_new_tokens,
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
    ):
        if inference_args:
            logger.warning("Overriding the default inference arguments with the provided arguments in .inference()")
            sampling_params = self._parse_args_to_sampling_params(inference_args)
        else:
            sampling_params = self.sampling_params

        model_input = model.prepare_inputs_for_inference(
            dataset=dataset,
            apply_chat_template=self.inferencer_args.apply_chat_template,
            inference_engine="sglang",
        )

        outputs = model.inference(
            inputs=model_input,
            sampling_params=sampling_params,
            release_gpu=release_gpu,
            inference_engine="sglang",
            gpu_memory_utilization=self.inferencer_args.inference_gpu_memory_utilization,
            tensor_parallel_size=self.inferencer_args.inference_tensor_parallel_size,
        )

        if self.inferencer_args.save_results:
            self.save_inference_results(outputs, self.inferencer_args.results_path)

        return outputs

    def save_inference_results(
        self,
        outputs: Union[list[list[str]], list[list[list[int]]]],
        save_file_path: str,
    ):
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)

        logger.info(f"Inference results are saved to {save_file_path}.")

    def load_inference_results(
        self,
        results_path: str,
    ) -> Union[list[list[str]], list[list[list[int]]]]:
        with open(results_path) as f:
            results = json.load(f)

        return results
