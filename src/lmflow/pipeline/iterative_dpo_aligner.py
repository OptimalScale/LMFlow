from lmflow.pipeline.base_aligner import BaseAligner


class IterativeDPOAligner(BaseAligner):
    def __init__(self, model_args, reward_model_args, data_args, aligner_args):
        self.model_args = model_args
        self.reward_model_args = reward_model_args
        self.data_args = data_args
        self.aligner_args = aligner_args


    def align(self, model, dataset, reward_model):
        # step0. initialization
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

