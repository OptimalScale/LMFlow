#!/usr/bin/env python
# coding=utf-8
"""
A model maps "text_only" data to float.
"""

from lmflow.models.regression_model import RegressionModel
from lmflow.datasets.dataset import Dataset


class TextRegressionModel(RegressionModel):
    r"""
    Initializes a TextRegressionModel instance.

    Parameters
    ------------

    model_args : 
        Model arguments such as model name, path, revision, etc.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.    
    """

    def __init__(
        self,
        model_args,
        *args,
        **kwargs
    ):
        """
        Initializes a TextRegressionModel instance.
        :param model_args: dictionary with model arguments such as model name, path, revision, etc.
        """
        self.inference_func = None


    def register_inference_function(self, inference_func):
        """
        Registers a regression function.
        """
        self.inference_func = inference_func


    def inference(self, inputs: Dataset):
        """
        Gets regression results of a given dataset.

        :inputs: Dataset object, only accept type "text_only".
        """
        if self.inference_func is not None:
            return self.inference_func(inputs)
        else:
            pass
