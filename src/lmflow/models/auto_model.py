#!/usr/bin/env python
# coding=utf-8
"""Automatically get correct model type.
"""

from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.models.text_regression_model import TextRegressionModel
from lmflow.models.hf_encoder_decoder_model import HFEncoderDecoderModel

class AutoModel:

    @classmethod
    def get_model(self, model_args, *args, **kwargs):
        arch_type = model_args.arch_type
        if arch_type == "decoder_only":
            return HFDecoderModel(model_args, *args, **kwargs)
        elif arch_type == "text_regression":
            return TextRegressionModel(model_args, *args, **kwargs)
        elif arch_type == "encoder_decoder":
            return HFEncoderDecoderModel(model_args, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"model architecture type \"{arch_type}\" is not supported"
            )
