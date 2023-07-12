#!/usr/bin/env python
# coding=utf-8
# TODO update the doc

import copy
import logging
import torch
from typing import List, Optional, Union

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Config,
    AutoModelForCausalLM
)

from .base_model import BaseModel

class CustomAutoVision2SeqModel(Blip2ForConditionalGeneration, BaseModel):
    def __init__(self, config: Blip2Config):
        Blip2ForConditionalGeneration.__init__(self, config)


    def vision_model_from_pretrained(self, pretrained_path):
        self.vision_model = self.vision_model.from_pretrained(
                                pretrained_path,
                                config=self.config.vision_config)
    def qformer_from_pretrained(self, pretrained_path):
        self.qformer = self.qformer.from_pretrained(
                                pretrained_path,
                                config=self.config.qformer_config)
        print(self.qformer.encoder.layer[11].output_query.dense.weight.mean())

    def language_model_from_pretrained(self, pretrained_path, low_resource=False):
        # TODO remove the low resource related loading in the future
        if low_resource:
            kwargs = dict(
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            kwargs = {}
        self.language_model = AutoModelForCausalLM.from_pretrained(
                                pretrained_path,
                                config=self.config.text_config,
                                **kwargs)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_token_indexes: Optional[List] = [0],
        one_sample_multiple_images: Optional[bool] = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
            image_token_indexes (bool, *optional*):
                The index for inserting the image tokens.
            one_sample_multiple_images: (bool, *optional*):
                The flag for inference that the input batch size is 1 and contain multiple images.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        if not one_sample_multiple_images:
            batch_size = pixel_values.shape[0]
        else:
            batch_size = 1
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(language_attention_mask.device)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(language_model_inputs.device)
        # concatenate the text embeddings with image embeddings
        inputs_embeds_with_images = []
        attention_mask_with_images = []
        # currently we only support with one image
        start_index, end_index = 0, 0
        import pdb; pdb.set_trace()
        assert len(image_token_indexes) == pixel_values.shape[0]
        # token format: (# text, # image)xN, # text
        for idx, image_token_index in enumerate(image_token_indexes):
            end_index += image_token_index
            inputs_embeds_with_images.append(
                inputs_embeds[:, start_index:end_index])
            inputs_embeds_with_images.append(language_model_inputs[idx][None])
            attention_mask_with_images.append(
                attention_mask[:, start_index:end_index])
            attention_mask_with_images.append(language_attention_mask[idx][None])
            start_index = end_index
    
        inputs_embeds_with_images.append(inputs_embeds[:, image_token_indexes[-1]:])
        inputs_embeds = torch.cat(inputs_embeds_with_images, dim=1)
        attention_mask_with_images.append(attention_mask[:, image_token_indexes[-1]:])
        attention_mask = torch.cat(attention_mask_with_images, dim=1)
        # comebine the embeds
        inputs_embeds = inputs_embeds.to(self.language_model.lm_head.weight.dtype)
        attention_mask = attention_mask.to(self.language_model.lm_head.weight.dtype)
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs
