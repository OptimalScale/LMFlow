from typing import List, Tuple

import numpy as np
import pytest
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

from lmflow.args import InferencerArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.sglang_inferencer import SGLangInferencer

from tests.datasets.conftest import dataset_inference_conversation_batch


@pytest.fixture
def sglang_test_model_args() -> ModelArguments:
    return ModelArguments(model_name_or_path="Qwen/Qwen3-4B-Instruct-2507")


@pytest.fixture
def sglang_test_inferencer_args() -> InferencerArguments:
    return InferencerArguments(
        inference_engine="sglang",
        inference_gpu_memory_utilization=0.8,
        num_output_sequences=2,
        enable_deterministic_inference=True,
        attention_backend="fa3",
        return_logprob=True,
    )


def test_sglang_inferencer(
    dataset_inference_conversation_batch: Dataset,
    sglang_test_model_args: ModelArguments,
    sglang_test_inferencer_args: InferencerArguments,
):
    def parse_logprob(logprob_list: List[Tuple[float, int, None]]) -> List[float]:
        return np.array([logprob for logprob, _, _ in logprob_list])

    model = HFDecoderModel(model_args=sglang_test_model_args)
    sglang_inferencer = SGLangInferencer(
        data_args=dataset_inference_conversation_batch.data_args,
        model_args=sglang_test_model_args,
        inferencer_args=sglang_test_inferencer_args,
    )
    res = sglang_inferencer.inference(
        model=model,
        dataset=dataset_inference_conversation_batch,
        release_gpu=True,
    )
    assert len(res) == 4
    assert res[0]["input"] == dataset_inference_conversation_batch.backend_dataset[0]["templated"]
    assert res[1]["input"] == dataset_inference_conversation_batch.backend_dataset[0]["templated"]
    assert res[2]["input"] == dataset_inference_conversation_batch.backend_dataset[1]["templated"]
    assert res[3]["input"] == dataset_inference_conversation_batch.backend_dataset[1]["templated"]

    # test consistency
    sgl_server_args = ServerArgs(
        model_path=sglang_test_model_args.model_name_or_path,
        mem_fraction_static=sglang_test_inferencer_args.inference_gpu_memory_utilization,
        tp_size=sglang_test_inferencer_args.inference_tensor_parallel_size,
        enable_deterministic_inference=sglang_test_inferencer_args.enable_deterministic_inference,
        attention_backend=sglang_test_inferencer_args.attention_backend,
    )
    llm = Engine(server_args=sgl_server_args)
    model_input = [
        sample for sample in dataset_inference_conversation_batch.backend_dataset['templated'] 
        for _ in range(sglang_test_inferencer_args.num_output_sequences)
    ]
    sglang_outputs = llm.generate(
        prompt=model_input,
        sampling_params=sglang_inferencer.sampling_params.copy().update({"n": 1}),
        return_logprob=sglang_test_inferencer_args.return_logprob,
    )
    logprobs_lmflow = [parse_logprob(x["meta_info"]["output_token_logprobs"]) for x in res]
    logprobs_sglang = [parse_logprob(x["meta_info"]["output_token_logprobs"]) for x in sglang_outputs]
    
    assert all(
        np.allclose(logprobs_lmflow, logprobs_sglang, atol=1e-10) 
        for logprobs_lmflow, logprobs_sglang in zip(logprobs_lmflow, logprobs_sglang)
    )
