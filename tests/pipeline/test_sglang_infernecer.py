import pytest

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
    )


def test_sglang_inferencer(
    dataset_inference_conversation_batch: Dataset,
    sglang_test_model_args: ModelArguments,
    sglang_test_inferencer_args: InferencerArguments,
):
    model = HFDecoderModel(model_args=sglang_test_model_args)
    sglang_inferencer = SGLangInferencer(
        data_args=dataset_inference_conversation_batch.data_args,
        model_args=sglang_test_model_args,
        inferencer_args=sglang_test_inferencer_args,
    )
    res = sglang_inferencer.inference(
        model=model,
        dataset=dataset_inference_conversation_batch,
    )
    assert len(res) == 4
    assert res[0]["input"] == dataset_inference_conversation_batch.backend_dataset[0]["templated"]
    assert res[1]["input"] == dataset_inference_conversation_batch.backend_dataset[0]["templated"]
    assert res[2]["input"] == dataset_inference_conversation_batch.backend_dataset[1]["templated"]
    assert res[3]["input"] == dataset_inference_conversation_batch.backend_dataset[1]["templated"]
