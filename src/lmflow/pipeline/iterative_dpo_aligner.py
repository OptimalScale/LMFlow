import logging

from lmflow.args import ModelArguments, DatasetArguments, IterativeDPOAlignerArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.pipeline.base_aligner import BaseAligner
from lmflow.pipeline.utils.collections import check_homogeneity


logger = logging.getLogger(__name__)


class IterativeDPOAligner(BaseAligner):
    def __init__(
        self, 
        model_args: ModelArguments,
        reward_model_args: ModelArguments,
        data_args: DatasetArguments,
        aligner_args: IterativeDPOAlignerArguments,
    ):
        self.model_args = model_args
        self.reward_model_args = reward_model_args
        self.data_args = data_args
        self.aligner_args = aligner_args


    def align(
        self, 
        model_args: ModelArguments,
        data_args: DatasetArguments,
        reward_model_args: ModelArguments,
    ) -> HFDecoderModel:
        # step0. isnitialization fixed objects, preprocessing
        # step1. align
        is_homogenenous = check_homogeneity([model_args, reward_model_args])
        if is_homogenenous:
            logging.info(
                "SFT model and reward model have the same tokenizer, "
                "passing intermediate results in tensors."
            )
            aligned_model = self._align_homogeneous(model_args, data_args, reward_model_args)
        else:
            logging.info(
                "SFT model and reward model have different tokenizers, "
                "saving intermediate results in string format."
            )
            aligned_model = self._align_heterogeneous(model_args, data_args, reward_model_args)
    
    
    def _align_homogeneous(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        reward_model_args: ModelArguments,
    ):
        # step0. init
        # step1. do generation
        # step2. do scoring
        # step3. do train
        # step4. return & save
        pass
    
    
    def _align_heterogeneous(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        reward_model_args: ModelArguments,
    ):
        # step0. init
        # step1. do generation
        # step2. do scoring
        # step3. do train
        # step4. return & save
        pass
    
    
    def _do_generation(self):
        pass
    
    
    def _do_scoring(self):
        pass
    
    
    def _do_train(self):
        pass

