#!/usr/bin/env python
# coding=utf-8
"""Return a pipeline automatically based on its name.
"""
import pkg_resources

def is_package_version_at_least(package_name, min_version):
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        if (pkg_resources.parse_version(package_version)
                < pkg_resources.parse_version(min_version)):
            return False
    except pkg_resources.DistributionNotFound:
        return False
    return True

from lmflow.pipeline.evaluator import Evaluator
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer
from lmflow.pipeline.vllm_inferencer import VLLMInferencer
from lmflow.pipeline.dpo_aligner import DPOAligner
from lmflow.pipeline.dpov2_aligner import DPOv2Aligner
from lmflow.pipeline.rm_tuner import RewardModelTuner
from lmflow.pipeline.rm_inferencer import RewardModelInferencer
PIPELINE_MAPPING = {
    "evaluator": Evaluator,
    "finetuner": Finetuner,
    "inferencer": Inferencer,
    "vllm_inferencer": VLLMInferencer,
    "rm_inferencer": RewardModelInferencer,
    "dpo_aligner": DPOAligner,
    "dpov2_aligner": DPOv2Aligner,
    "rm_tuner": RewardModelTuner,
}

if not is_package_version_at_least('transformers', '4.35.0'):
    from lmflow.pipeline.raft_aligner import RaftAligner
    PIPELINE_MAPPING['raft_aligner'] = RaftAligner


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
