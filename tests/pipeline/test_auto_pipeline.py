import unittest

from lmflow.args import DatasetArguments
from lmflow.args import EvaluatorArguments
from lmflow.args import FinetunerArguments
from lmflow.args import InferencerArguments
from lmflow.args import ModelArguments
from lmflow.args import RaftAlignerArguments
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.pipeline.evaluator import Evaluator
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer
from lmflow.pipeline.raft_aligner import RaftAligner

MODEL_NAME = "gpt2"


class AutoPipelineTest(unittest.TestCase):

    def test_get_evaluator_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        evaluator_args = EvaluatorArguments()
        pipeline = AutoPipeline.get_pipeline(
            "evaluator", model_args, dataset_args, evaluator_args)

        self.assertTrue(isinstance(pipeline, Evaluator))

    def test_get_finetuner_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        finetuner_args = FinetunerArguments(output_dir="~/tmp")
        pipeline = AutoPipeline.get_pipeline(
            "finetuner", model_args, dataset_args, finetuner_args)

        self.assertTrue(isinstance(pipeline, Finetuner))

    def test_get_inferencer_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        inferencer_args = InferencerArguments()
        pipeline = AutoPipeline.get_pipeline(
            "inferencer", model_args, dataset_args, inferencer_args)

        self.assertTrue(isinstance(pipeline, Inferencer))

    def test_get_raft_aligner_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        raft_aligner_args = RaftAlignerArguments(output_dir="~/tmp")
        pipeline = AutoPipeline.get_pipeline(
            "raft_aligner", model_args, dataset_args, raft_aligner_args)

        self.assertTrue(isinstance(pipeline, RaftAligner))

    def test_get_unsupported_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()

        with self.assertRaisesRegex(NotImplementedError, "Pipeline \"unsupported\" is not supported"):
            pipeline = AutoPipeline.get_pipeline(
                "unsupported", model_args, dataset_args, None)
