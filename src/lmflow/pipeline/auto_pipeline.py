#!/usr/bin/env python
# coding=utf-8
"""Return a pipeline automatically based on its name.
"""

from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.evaluator import Evaluator
from lmflow.pipeline.inferencer import Inferencer


PIPELINE_MAPPING = {
    "seq2seq_finetuner": Finetuner,
    "finetuner": Finetuner,
    "evaluator": Evaluator,
    "inferencer": Inferencer,
}


class AutoPipeline:
    """ 
    The class designed to return a pipeline automatically based on its name.
    """
    @classmethod
    def get_pipeline(self,
        pipeline_name,
        model_args,
        data_args,
        pipeline_args,
        *args,
        **kwargs
    ):
        if pipeline_name not in PIPELINE_MAPPING:
            raise NotImplementedError(
                f'Pipeline "{pipeline_name}" is not supported'
            )

        pipeline = PIPELINE_MAPPING[pipeline_name](
            model_args,
            data_args,
            pipeline_args,
            *args,
            **kwargs
        )
        return pipeline
