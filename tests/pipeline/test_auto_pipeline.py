import unittest

from lmflow.args import DatasetArguments, EvaluatorArguments, FinetunerArguments, InferencerArguments, ModelArguments
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.pipeline.evaluator import Evaluator
from lmflow.pipeline.finetuner import Finetuner
from lmflow.pipeline.inferencer import Inferencer

MODEL_NAME = "gpt2"


class AutoPipelineTest(unittest.TestCase):
    def test_get_evaluator_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        evaluator_args = EvaluatorArguments()
        pipeline = AutoPipeline.get_pipeline("evaluator", model_args, dataset_args, evaluator_args)

        self.assertTrue(isinstance(pipeline, Evaluator))

    def test_get_finetuner_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        finetuner_args = FinetunerArguments(output_dir="~/tmp")
        pipeline = AutoPipeline.get_pipeline("finetuner", model_args, dataset_args, finetuner_args)

        self.assertTrue(isinstance(pipeline, Finetuner))

    def test_get_inferencer_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()
        inferencer_args = InferencerArguments()
        pipeline = AutoPipeline.get_pipeline("inferencer", model_args, dataset_args, inferencer_args)

        self.assertTrue(isinstance(pipeline, Inferencer))

    def test_get_unsupported_pipeline(self):
        model_args = ModelArguments(model_name_or_path=MODEL_NAME)
        dataset_args = DatasetArguments()

        with self.assertRaisesRegex(NotImplementedError, 'Pipeline "unsupported" is not supported'):
            pipeline = AutoPipeline.get_pipeline("unsupported", model_args, dataset_args, None)
