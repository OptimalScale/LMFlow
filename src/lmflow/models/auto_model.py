#!/usr/bin/env python
# coding=utf-8
"""Automatically get correct model type.
"""

from lmflow.models.hf_decoder_model import HFDecoderModel


class AutoModel:

    @classmethod
    def get_model(self, model_args, *args, **kwargs):
        # TODO (add new models)
        return HFDecoderModel(model_args, *args, **kwargs)
