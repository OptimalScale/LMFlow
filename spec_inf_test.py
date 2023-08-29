from lmflow.args import InferencerArguments
from lmflow.args import ModelArguments
from lmflow.args import RaftAlignerArguments
from lmflow.args import DatasetArguments
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.pipeline.evaluator import Evaluator
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer
from lmflow.pipeline.raft_aligner import RaftAligner


from src.lmflow.pipeline.inferencer import SpeculativeInferencer

model_args = ModelArguments(model_name_or_path='gpt2-large')
draft_model_args = ModelArguments(model_name_or_path='gpt2')
inferencer_args = InferencerArguments()
data_args = DatasetArguments()
specinf = SpeculativeInferencer(model_args, draft_model_args, data_args, inferencer_args)
