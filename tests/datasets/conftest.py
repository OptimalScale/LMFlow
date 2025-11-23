import pytest

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset


@pytest.fixture
def dataset_inference_conversation() -> Dataset:
    dataset = Dataset(DatasetArguments(dataset_path=None))
    dataset = dataset.from_dict(
        {"type": "conversation", "instances": [{"messages": [{"role": "user", "content": "Hello, how are you?"}]}]}
    )
    return dataset

@pytest.fixture
def dataset_inference_conversation_batch() -> Dataset:
    dataset = Dataset(DatasetArguments(dataset_path=None))
    dataset = dataset.from_dict(
        {
            "type": "conversation", 
            "instances": [
                {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
                {"messages": [{"role": "user", "content": "What's the capital of France?"}]},
            ]
        }
    )
    return dataset