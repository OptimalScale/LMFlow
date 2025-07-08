#!/usr/bin/env python
"""BaseTuner: a subclass of BasePipeline."""

from abc import abstractmethod

from lmflow.pipeline.base_pipeline import BasePipeline


class BaseTuner(BasePipeline):
    """A subclass of BasePipeline which is tunable."""

    def __init__(self, *args, **kwargs):
        pass

    def _check_if_tunable(self, model, dataset):
        # TODO: check if the model is tunable and dataset is compatible
        pass

    @abstractmethod
    def tune(self, model, dataset):
        raise NotImplementedError(".tune is not implemented")
