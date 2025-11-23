import pytest

from lmflow.args import DatasetArguments, InferencerArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.sglang_inferencer import SGLangInferencer

from tests.datasets.conftest import dataset_inference_conversation


@pytest.fixture
def sglang_test_model_args() -> ModelArguments:
    return ModelArguments(model_name_or_path="Qwen/Qwen3-4B-Instruct-2507")

@pytest.fixture
def sglang_test_inferencer_args() -> InferencerArguments:
    return InferencerArguments(inference_engine="sglang")

if __name__ == "__main__":
    def test_sglang_inferencer(
        dataset_inference_conversation: Dataset,
        sglang_test_model_args: ModelArguments,
        sglang_test_inferencer_args: InferencerArguments
    ):
        model = HFDecoderModel(model_args=sglang_test_model_args)
        sglang_inferencer = SGLangInferencer(
            data_args=dataset_inference_conversation.data_args, 
            model_args=sglang_test_model_args, 
            inferencer_args=sglang_test_inferencer_args
        )
        sglang_inferencer.inference(
            model=model,
            dataset=dataset_inference_conversation,
        )
    test_sglang_inferencer()