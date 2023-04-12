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
        self.regression_func = None


    def register_regression_function(self, regression_func):
        """
        Registers a regression function.
        """
        self.regression_func = regression_func


    def get_regression(self, dataset: Dataset):
        """
        Gets regression results of a given dataset.

        :dataset: Dataset object, only accept type "text_only".
        """
        if self.regression_func is not None:
            return self.regression_func(dataset)
        else:
            pass
