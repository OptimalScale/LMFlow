#!/usr/bin/env python
# coding=utf-8
""" BaseTuner: a subclass of BasePipeline.
"""

from lmflow.pipeline.base_pipeline import BasePipeline


class BaseTuner(BasePipeline):
    """ A subclass of BasePipeline which is tunable.
    """
    def __init__(self, *args, **kwargs):
        pass

    def _check_if_tunable(self, model, dataset):
        # TODO: check if the model is tunable and dataset is compatible
        pass

    def tune(self, model, dataset):
        raise NotImplementedError(".tune is not implemented")
