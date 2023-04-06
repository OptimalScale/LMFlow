#!/usr/bin/env python
# coding=utf-8
"""Automatically get correct model type.
"""

from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.models.hf_encoder_decoder_model import HFEncoderDecoderModel

class AutoModel:

    @classmethod
    def get_model(self, model_args, *args, **kwargs):
        # TODO (add new models)
        if model_args.arch_type == "encoder_decoder":
            return HFEncoderDecoderModel(model_args, *args, **kwargs)
        elif model_args.arch_type == "decoder_only":
            return HFDecoderModel(model_args, *args, **kwargs)
