#!/usr/bin/env python
# coding=utf-8
"""Return a pipeline automatically based on its name.
"""
from lmflow.utils.versioning import (
    is_package_version_at_least, 
    is_vllm_available, 
    is_trl_available, 
    is_ray_available
)

from lmflow.pipeline.evaluator import Evaluator
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer
from lmflow.pipeline.rm_tuner import RewardModelTuner
from lmflow.pipeline.rm_inferencer import RewardModelInferencer
PIPELINE_MAPPING = {
    "evaluator": Evaluator,
    "finetuner": Finetuner,
    "inferencer": Inferencer,
    "rm_inferencer": RewardModelInferencer,
    "rm_tuner": RewardModelTuner,
}
PIPELINE_NEEDS_EXTRAS = []

if not is_package_version_at_least('transformers', '4.35.0'):
    from lmflow.pipeline.raft_aligner import RaftAligner
    PIPELINE_MAPPING['raft_aligner'] = RaftAligner
else:
    PIPELINE_NEEDS_EXTRAS.append('raft_aligner')
    
if is_vllm_available():
    from lmflow.pipeline.vllm_inferencer import VLLMInferencer
    PIPELINE_MAPPING['vllm_inferencer'] = VLLMInferencer
else:
    PIPELINE_NEEDS_EXTRAS.append('vllm_inferencer')
    
if is_trl_available():
    from lmflow.pipeline.dpo_aligner import DPOAligner
    from lmflow.pipeline.dpov2_aligner import DPOv2Aligner
    PIPELINE_MAPPING['dpo_aligner'] = DPOAligner
    PIPELINE_MAPPING['dpov2_aligner'] = DPOv2Aligner
else:
    PIPELINE_NEEDS_EXTRAS.extend(['dpo_aligner', 'dpov2_aligner'])
    
if is_vllm_available() and is_trl_available() and is_ray_available():
    from lmflow.pipeline.iterative_dpo_aligner import IterativeDPOAligner
    PIPELINE_MAPPING['iterative_dpo_aligner'] = IterativeDPOAligner
else:
    PIPELINE_NEEDS_EXTRAS.append('iterative_dpo_aligner')


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
            if pipeline_name in PIPELINE_NEEDS_EXTRAS:
                raise NotImplementedError(
                    f'Please install the necessary dependencies '
                    f'to use pipeline "{pipeline_name}"'
                )
                
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
