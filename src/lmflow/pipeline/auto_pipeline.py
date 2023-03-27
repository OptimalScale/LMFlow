#!/usr/bin/env python
# coding=utf-8
"""Return Finetuner or Inferencer pipeline automatically.
"""

from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer


PIPELINE_MAPPING = {
    "finetuner": Finetuner,
    "inferencer": Inferencer,
}


class AutoPipeline:
    """ 
    The class designed to return Finetuner or Inferencer pipeline automatically.
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
