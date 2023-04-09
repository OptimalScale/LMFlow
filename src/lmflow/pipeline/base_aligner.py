#!/usr/bin/env python
# coding=utf-8
""" BaseTuner: a subclass of BasePipeline.
"""

from lmflow.pipeline.base_pipeline import BasePipeline


class BaseAligner(BasePipeline):
    """ A subclass of BasePipeline which is alignable.
    """
    def __init__(self, *args, **kwargs):
        pass

    def _check_if_alignable(self, model, dataset, reward_model):
        # TODO: check if the model is alignable and dataset is compatible
        # TODO: add reward_model
        pass

    def align(self, model, dataset, reward_model):
        raise NotImplementedError(".align is not implemented")
