"""This Python code defines a class Dataset with methods for initializing, loading,
and manipulating datasets from different backends such as Hugging Face and JSON.
 
The `Dataset` class includes methods for loading datasets from a dictionary and a Hugging
Face dataset, mapping datasets, and retrieving the backend dataset and arguments.
"""
from lmflow.utils.versioning import is_multimodal_available


from lmflow.datasets.dataset import Dataset
if is_multimodal_available():
    from lmflow.datasets.multi_modal_dataset import CustomMultiModalDataset
