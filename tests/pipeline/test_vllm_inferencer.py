import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lmflow.args import DatasetArguments, InferencerArguments, ModelArguments
from lmflow.utils.protocol import DataProto


@pytest.fixture
def model_args():
    return ModelArguments(model_name_or_path="Qwen/Qwen2-0.5B")


@pytest.fixture
def inferencer_args():
    return InferencerArguments(
        inference_engine="vllm",
        inference_gpu_memory_utilization=0.8,
        num_output_sequences=2,
        temperature=1.0,
        max_new_tokens=128,
        top_p=0.95,
        top_k=50,
        random_seed=42,
        use_beam_search=False,
    )


@pytest.fixture
def data_args():
    return DatasetArguments(dataset_path=None)


class TestParseArgsToSamplingParams:
    """Test that _parse_args_to_sampling_params returns the correct dict."""

    def test_returns_dict(self, model_args, inferencer_args):
        with patch("lmflow.pipeline.vllm_inferencer.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.return_value = MagicMock(eos_token_id=151643)
            from lmflow.pipeline.vllm_inferencer import VLLMInferencer

            inferencer = VLLMInferencer(model_args, DatasetArguments(dataset_path=None), inferencer_args)
            params = inferencer.sampling_params

        assert isinstance(params, dict)
        assert set(params.keys()) == {"n", "temperature", "max_new_tokens", "seed", "top_p", "top_k", "stop_token_ids"}

    def test_values_match_args(self, model_args, inferencer_args):
        with patch("lmflow.pipeline.vllm_inferencer.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.return_value = MagicMock(eos_token_id=151643)
            from lmflow.pipeline.vllm_inferencer import VLLMInferencer

            inferencer = VLLMInferencer(model_args, DatasetArguments(dataset_path=None), inferencer_args)
            params = inferencer.sampling_params

        assert params["n"] == 2
        assert params["max_new_tokens"] == 128
        assert params["seed"] == 42
        assert params["top_p"] == 0.95
        assert params["top_k"] == 50
        assert abs(params["temperature"] - 1.0) < 1e-4
        assert 151643 in params["stop_token_ids"]

    def test_override_with_inference_args(self, model_args, inferencer_args):
        with patch("lmflow.pipeline.vllm_inferencer.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.return_value = MagicMock(eos_token_id=151643)
            from lmflow.pipeline.vllm_inferencer import VLLMInferencer

            inferencer = VLLMInferencer(model_args, DatasetArguments(dataset_path=None), inferencer_args)

        override_args = InferencerArguments(
            inference_engine="vllm",
            temperature=0.5,
            max_new_tokens=256,
            num_output_sequences=4,
        )
        new_params = inferencer._parse_args_to_sampling_params(override_args)
        assert new_params["max_new_tokens"] == 256
        assert new_params["n"] == 4
        assert abs(new_params["temperature"] - 0.5) < 1e-4


class TestDataProtoSaveLoad:
    """Test DataProto pickle round-trip used by save/load_inference_results."""

    def test_roundtrip(self):
        proto = DataProto.from_single_dict(
            data={"inputs": np.array(["Hello", "World"]), "outputs": np.array(["Hi", "Earth"])},
            meta_info={"sampling_params": {"n": 1, "temperature": 1.0}},
        )
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        proto.save_to_disk(path)
        loaded = DataProto.load_from_disk(path)

        assert len(loaded) == 2
        assert list(loaded.non_tensor_batch["inputs"]) == ["Hello", "World"]
        assert list(loaded.non_tensor_batch["outputs"]) == ["Hi", "Earth"]
        assert loaded.meta_info["sampling_params"]["n"] == 1

    def test_save_load_uses_dir(self):
        """VLLMInferencer saves inference_results.pkl inside the given directory."""
        with patch("lmflow.pipeline.vllm_inferencer.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.return_value = MagicMock(eos_token_id=0)
            from lmflow.pipeline.vllm_inferencer import VLLMInferencer

            args = InferencerArguments(inference_engine="vllm")
            inferencer = VLLMInferencer(
                ModelArguments(model_name_or_path="dummy"), DatasetArguments(dataset_path=None), args
            )

        proto = DataProto.from_single_dict(data={"inputs": np.array(["a"])})
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = f"{tmpdir}/results"
            os.makedirs(results_dir)
            inferencer.save_inference_results(proto, results_dir)
            assert os.path.exists(os.path.join(results_dir, "inference_results.pkl"))
            loaded = inferencer.load_inference_results(results_dir)
            assert len(loaded) == 1


class TestPrepareInputsDataProto:
    """Test that prepare_inputs_for_inference creates a proper DataProto for vllm."""

    def test_creates_dataproto_with_repeat(self):
        """Simulate what prepare_inputs_for_inference does for vllm."""
        prompts = ["prompt_a", "prompt_b"]
        sampling_params = {"n": 3, "temperature": 1.0}

        inference_inputs = np.array(prompts)
        proto = DataProto.from_single_dict(
            data={"inputs": inference_inputs},
            meta_info={"sampling_params": {**sampling_params, "n": 1}, "actual_n_rollouts": sampling_params["n"]},
        )
        proto = proto.repeat(sampling_params["n"])

        # 2 prompts * n=3 = 6 rows
        assert len(proto) == 6
        assert proto.meta_info["sampling_params"]["n"] == 1
        assert proto.meta_info["actual_n_rollouts"] == 3

        inputs_list = proto.non_tensor_batch["inputs"].tolist()
        # repeat interleaves: [a, a, a, b, b, b]
        assert inputs_list == ["prompt_a"] * 3 + ["prompt_b"] * 3


vllm = pytest.importorskip("vllm")

from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.vllm_inferencer import VLLMInferencer
from tests.datasets.conftest import dataset_inference_conversation_batch  # noqa: F401


@pytest.fixture
def vllm_test_model_args() -> ModelArguments:
    return ModelArguments(model_name_or_path="Qwen/Qwen3-0.6B")


@pytest.fixture
def vllm_test_inferencer_args() -> InferencerArguments:
    return InferencerArguments(
        inference_engine="vllm",
        inference_gpu_memory_utilization=0.8,
        num_output_sequences=2,
        temperature=1.0,
        max_new_tokens=64,
        top_p=0.95,
        random_seed=42,
    )


@pytest.mark.gpu
def test_vllm_inferencer(
    dataset_inference_conversation_batch: Dataset,  # noqa: F811
    vllm_test_model_args: ModelArguments,
    vllm_test_inferencer_args: InferencerArguments,
):
    model = HFDecoderModel(model_args=vllm_test_model_args)
    inferencer = VLLMInferencer(
        data_args=dataset_inference_conversation_batch.data_args,
        model_args=vllm_test_model_args,
        inferencer_args=vllm_test_inferencer_args,
    )
    res = inferencer.inference(
        model=model,
        dataset=dataset_inference_conversation_batch,
        release_gpu=True,
    )

    # DataProto structure checks
    assert isinstance(res, DataProto)

    # 2 conversations * n=2 = 4 rows
    assert len(res) == 4

    # Has inputs and outputs in non_tensor_batch
    assert "inputs" in res.non_tensor_batch
    assert "outputs" in res.non_tensor_batch
    assert len(res.non_tensor_batch["inputs"]) == 4
    assert len(res.non_tensor_batch["outputs"]) == 4

    # Each output should be a non-empty string
    for output in res.non_tensor_batch["outputs"]:
        assert isinstance(output, str)
        assert len(output) > 0

    # Sampling params in meta_info
    assert "sampling_params" in res.meta_info
    assert res.meta_info["sampling_params"]["n"] == 1

    # Inputs repeat pattern: [conv1, conv1, conv2, conv2]
    inputs = res.non_tensor_batch["inputs"].tolist()
    assert inputs[0] == inputs[1]
    assert inputs[2] == inputs[3]
    assert inputs[0] != inputs[2]


@pytest.mark.gpu
def test_vllm_inferencer_save_load(
    dataset_inference_conversation_batch: Dataset,  # noqa: F811
    vllm_test_model_args: ModelArguments,
    vllm_test_inferencer_args: InferencerArguments,
):
    model = HFDecoderModel(model_args=vllm_test_model_args)
    inferencer = VLLMInferencer(
        data_args=dataset_inference_conversation_batch.data_args,
        model_args=vllm_test_model_args,
        inferencer_args=vllm_test_inferencer_args,
    )
    res = inferencer.inference(
        model=model,
        dataset=dataset_inference_conversation_batch,
        release_gpu=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        inferencer.save_inference_results(res, results_dir)
        loaded = inferencer.load_inference_results(results_dir)

        assert len(loaded) == len(res)
        assert list(loaded.non_tensor_batch["inputs"]) == list(res.non_tensor_batch["inputs"])
        assert list(loaded.non_tensor_batch["outputs"]) == list(res.non_tensor_batch["outputs"])
