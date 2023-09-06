from lmflow.args import InferencerArguments
from lmflow.args import ModelArguments
from lmflow.args import DatasetArguments
from lmflow.models import hf_decoder_model
from src.lmflow.pipeline.inferencer import SpeculativeInferencer
import logging

logging.basicConfig(level=logging.DEBUG)

model_args = ModelArguments(model_name_or_path='gpt2-large')
model = hf_decoder_model.HFDecoderModel(model_args)
draft_model_args = ModelArguments(model_name_or_path='gpt2')
draft_model = hf_decoder_model.HFDecoderModel(draft_model_args)

inferencer_args = InferencerArguments()
data_args = DatasetArguments()

specinf = SpeculativeInferencer(model_args, draft_model_args, data_args, inferencer_args)

specinf.inference(model, draft_model, 'Hello, how are you', gamma=3, max_new_tokens=10)