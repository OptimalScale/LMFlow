import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.dpov2_aligner import DPOv2Aligner
from lmflow.pipeline.rm_inferencer import RewardModelInferencer
from lmflow.pipeline.vllm_inferencer import MemorySafeVLLMInferencer
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    IterativeDPOAlignerArguments
)

logger = logging.getLogger(__name__)


class IterativeDPOAligner:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        aligner_args:IterativeDPOAlignerArguments,
        ref_model_args: ModelArguments,
        reward_model_args: ModelArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = aligner_args
        self.ref_model_args = ref_model_args
        self.reward_model_args = reward_model_args
        self.workspace_path = Path(self.aligner_args.output_dir)
        self.iteration_path_dict = {}
        
        
    def align(
        self,
        model: HFDecoderModel,
        ref_model: HFDecoderModel,
        reward_model: HFTextRegressionModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        transform_dataset_in_place: bool=True,
    ):
        pass
    
    
    def _align_single_iteration(
        self,
        target_model_inferencer: MemorySafeVLLMInferencer,
        reward_model_inferencer: RewardModelInferencer,
        aligner: DPOv2Aligner,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        transform_dataset_in_place: bool=True,
    ):
        pass
    
    
    def do_target_model_inference(
        self,
        model: HFDecoderModel,
        dataset: Dataset,
        output_dir: Optional[str]=None,
    ):
        pass