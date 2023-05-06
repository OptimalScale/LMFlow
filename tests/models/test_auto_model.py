import unittest

from lmflow.args import ModelArguments
from lmflow.models.auto_model import AutoModel
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.models.text_regression_model import TextRegressionModel
from lmflow.models.hf_encoder_decoder_model import HFEncoderDecoderModel

MODEL_NAME = "gpt2"


class AutoModelTest(unittest.TestCase):

    def test_get_decoder_model(self):
        model_args = ModelArguments(
            arch_type="decoder_only", model_name_or_path=MODEL_NAME)
        model = AutoModel.get_model(model_args)
        self.assertTrue(isinstance(model, HFDecoderModel))


    # This unit test is commented out since the encoder decoder model has not been fully implemented
    '''
    def test_get_text_regression_model(self):
        model_args = ModelArguments(
            arch_type="text_regression", model_name_or_path=MODEL_NAME)
        model = AutoModel.get_model(model_args)
        self.assertTrue(isinstance(model, TextRegressionModel))
    '''


    # This unit test is commented out since the encoder decoder model has not been fully implemented
    '''
    def test_get_encoder_decoder(self):
        model_args = ModelArguments(
            arch_type="encoder_decoder", model_name_or_path=MODEL_NAME)
        model = AutoModel.get_model(model_args)
        self.assertTrue(isinstance(model, HFEncoderDecoderModel))
    '''


    def test_get_unsupported_model(self):
        model_args = ModelArguments(
            arch_type="unsupported model", model_name_or_path=MODEL_NAME)
        with self.assertRaises(NotImplementedError):
            model = AutoModel.get_model(model_args)
