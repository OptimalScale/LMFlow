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