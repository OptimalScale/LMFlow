#!/usr/bin/env python
# coding=utf-8
# TODO update the doc

import copy
import logging
import time
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    Blip2ForConditionalGeneration,
    Blip2Config,
    Blip2QFormerModel,
    Blip2VisionModel,
    Blip2PreTrainedModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.deepspeed import is_deepspeed_zero3_enabled

from lmflow.models.base_model import BaseModel
from lmflow.models.vision_encoder import build_vision_tower


class CustomAutoVision2SeqModel(Blip2ForConditionalGeneration, BaseModel):
    def __init__(self,
                 config: Blip2Config,
                 image_encoder_name_or_path=None,
                 qformer_name_or_path=None,
                 language_model_name_or_path=None,
                 low_resource=False,):
        '''
        TODO update the docs
        Args:
            config:
            # the below varaible are used to overwrite the model in config
            image_encoder_name_or_path:
            qformer_name_or_path:
            language_model_name_or_path:
        Returns:
        '''
        super(Blip2PreTrainedModel, self).__init__(config)
        self.custom_vision_model = getattr(
            config, "custom_vision_model", False)
        self.with_qformer = getattr(config, "with_qformer", True)
        # vision model
        if self.custom_vision_model:
            # custom vision model means the vit model customized from llava.
            # vision_model_args = getattr(config, "vision_model_args", dict())
            self.vision_model = build_vision_tower(config)
            config.vision_config = self.vision_model.config
            self.image_processor = self.vision_model.image_processor
        elif image_encoder_name_or_path is not None:
            # use the model from transformers
            self.vision_model = AutoModel.from_pretrained(
                image_encoder_name_or_path)
            config.vision_config = self.vision_model.config
        else:
            # the default vit in Blip2
            self.vision_model = Blip2VisionModel(config.vision_config)
        if self.with_qformer:
            # check if with qformer, the blip series model use qformer
            # and the llava based models don't use qformer.
            if qformer_name_or_path is not None:
                self.query_tokens = nn.Parameter(
                    torch.zeros(1, config.num_query_tokens,
                                config.qformer_config.hidden_size))
                self.qformer = AutoModel.from_pretrained(
                    qformer_name_or_path)
            else:
                self.query_tokens = nn.Parameter(
                    torch.zeros(1, config.num_query_tokens,
                                config.qformer_config.hidden_size))
                self.qformer = Blip2QFormerModel(config.qformer_config)
        kwargs = dict()
        if language_model_name_or_path is not None:
            if low_resource:
                kwargs = dict(
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True)
            else:
                if not is_deepspeed_zero3_enabled:
                    kwargs = dict(device_map="auto",
                            torch_dtype=torch.float16)
            language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name_or_path, **kwargs)
            config.text_config = language_model.config
        else:
            if config.use_decoder_only_language_model:
                language_model = AutoModelForCausalLM.from_config(
                    config.text_config, **kwargs)
            else:
                language_model = AutoModelForSeq2SeqLM.from_config(
                    config.text_config, **kwargs)
        # Update _tied_weights_keys using the base model used.
        if getattr(language_model, "_tied_weights_keys", None) is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model
        if self.with_qformer:
            self.language_projection = nn.Linear(
                self.qformer.config.hidden_size,
                self.language_model.config.hidden_size)
        else:
            self.language_projection = nn.Linear(
                self.vision_model.hidden_size,
                self.language_model.config.hidden_size)
        if image_encoder_name_or_path is None and \
           language_model_name_or_path is None:
            self.post_init()
        # for deepspeed
        self.hidden_size = self.language_model.config.hidden_size
        self.config.hidden_size = self.language_model.config.hidden_size

    def get_backend_model(self):
        return self

    def vision_model_from_pretrained(self, pretrained_path):
        self.vision_model = self.vision_model.from_pretrained(
                                pretrained_path,
                                config=self.config.vision_config)

    def qformer_from_pretrained(self, pretrained_path):
        self.qformer = self.qformer.from_pretrained(
                                pretrained_path,
                                config=self.config.qformer_config)

    def language_model_from_pretrained(self,
                                       pretrained_path,
                                       low_resource=False,
                                       use_prompt_cache=False):
        # TODO remove the low resource related loading in the future
        self.use_prompt_cache = use_prompt_cache
        if low_resource:
            kwargs = dict(
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            kwargs = {}
        past_model_dim = self.language_model.model_dim
        self.language_model = AutoModelForCausalLM.from_pretrained(
                                pretrained_path,
                                config=self.config.text_config,
                                **kwargs)
        if self.config.text_config.hidden_size != past_model_dim:
            # should update the language projection layer
            in_channels = self.language_projection.in_features
            self.language_projection = nn.Linear(in_channels,
                                                 self.config.text_config.hidden_size,
                                                 bias=True)

    def vision_feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.vision_feature_select_layer]
        if self.select_vision_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif self.select_vision_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def register_prompt_cache(self, prompt_ids, prompt_keys_values):
        """
        Udpate the prompt id and embedding for reuse in the future

        Args:
            prompt_ids (torch.LongTensor): The id of the prompt.
            prompt_keys_values (torch.FloatTensor): The embedding of the prompt.

        Returns:
            None
        """
        self.prompt_ids = prompt_ids
        self.prompt_keys_values = prompt_keys_values
        self.with_prompt_cache = True

    def save_prompt_cache(self, path):
        """
        Save prompt embedding and id.

        Args:
            path: The path to save the prompt embedding and id.

        Returns:
            None
        """

        torch.save(
            dict(
                prompt_ids=self.prompt_ids,
                prompt_keys_values=self.prompt_keys_values
            ),
            path)

    def load_prompt_cache(self, path):
        """
        Load prompt embedding and id.
        Args:
            path: The path to load the prompt embedding and id.

        Returns:
            None
        """
        prompt_cache = torch.load(path)
        self.register_prompt_cache(prompt_cache["prompt_ids"],
                                   prompt_cache["prompt_keys_values"])

    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_token_indexes: Optional[List] = [0],
        one_sample_multiple_images: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if pixel_values is None and images is not None:
            pixel_values = images

        if not one_sample_multiple_images:
            batch_size = pixel_values.shape[0]
        else:
            batch_size = 1
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not self.custom_vision_model:
            # do the processing as blip2 and mini gpt-4;
            if past_key_values is not None and input_ids.shape[1] == 1:
                # no need to recompute the key values
                attention_mask = torch.ones((
                    attention_mask.shape[0],
                    past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device)
            else:
                image_embeds = self.vision_model(
                    pixel_values, return_dict=True).last_hidden_state
                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=image_embeds.device)
                if self.with_qformer:
                    query_tokens = self.query_tokens.expand(
                        image_embeds.shape[0], -1, -1)
                    query_outputs = self.qformer(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_attention_mask,
                        return_dict=True,
                    )
                else:
                    query_outputs = image_embeds
                query_output = query_outputs.last_hidden_state
                language_model_inputs = self.language_projection(query_output)
                inputs_embeds, attention_mask = \
                    self.processor_image_token_in_minigpt4(
                        input_ids,
                        language_model_inputs,
                        attention_mask,
                        image_token_indexes,
                        pixel_values,
                        batch_size)
                input_ids = None
        else:
            # do the processing in the vision model
            # language is the causallm model.
            # so use language model.model to do the embed_tokens
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
                self.vision_model.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask,
                    past_key_values, labels,
                    pixel_values,
                    self.language_projection,
                    self.language_model.model)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # TODO check how to generate the labels with image embeddings
        # print(input_ids, attention_mask)
        # if inputs_embeds is not None:
        #     print("input_embeds", inputs_embeds.shape)
        # attention_mask.shape, inputs_embeds.shape)
        # TODO remove this code by fixing the ddp training issue
        inputs_embeds = inputs_embeds.to(
            self.language_model.lm_head.weight.dtype)
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if labels is not None:
            logits = outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (shift_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def processor_image_token_in_minigpt4(self,
                                          input_ids,
                                          language_model_inputs,
                                          attention_mask,
                                          image_token_indexes,
                                          pixel_values,
                                          batch_size=1):
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(language_model_inputs.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(language_attention_mask.device)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(device=language_model_inputs.device)
        # concatenate the text embeddings with image embeddings
        inputs_embeds_with_images = []
        attention_mask_with_images = []
        # currently we only support with one image
        start_index, end_index = 0, 0
        assert len(image_token_indexes) == pixel_values.shape[0]
        # token format: (# text, # image)xN, # text

        for idx, image_token_index in enumerate(image_token_indexes):
            end_index += image_token_index
            inputs_embeds_with_images.append(
                inputs_embeds[:, start_index:end_index])
            inputs_embeds_with_images.append(language_model_inputs[idx][None])
            attention_mask_with_images.append(
                attention_mask[:, start_index:end_index])
            attention_mask_with_images.append(
                language_attention_mask[idx][None])
            start_index = end_index

        inputs_embeds_with_images.append(
            inputs_embeds[:, image_token_indexes[-1]:])
        inputs_embeds = torch.cat(
            inputs_embeds_with_images, dim=1)
        attention_mask_with_images.append(
            attention_mask[:, image_token_indexes[-1]:])
        attention_mask = torch.cat(attention_mask_with_images, dim=1)
        # comebine the embeds
        inputs_embeds = inputs_embeds.to(
            self.language_model.lm_head.weight.dtype)
        attention_mask = attention_mask.to(
            self.language_model.lm_head.weight.dtype)
        return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_token_indexes: Optional[List] = [0],
        one_sample_multiple_images: Optional[bool] = False,
        images: Optional[torch.LongTensor] = None,
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
        if pixel_values is None and images is not None:
            pixel_values = images

        if not one_sample_multiple_images:
            batch_size = pixel_values.shape[0]
        else:
            batch_size = 1
        if not self.custom_vision_model:
            # do the processing as blip2 and mini gpt-4;
            image_embeds = self.vision_model(
                pixel_values, return_dict=True).last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device)
            if self.with_qformer:
                query_tokens = self.query_tokens.expand(
                    image_embeds.shape[0], -1, -1)
                query_outputs = self.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
            else:
                query_outputs = image_embeds
            query_output = query_outputs.last_hidden_state
            language_model_inputs = self.language_projection(query_output)
            inputs_embeds, attention_mask = \
                self.processor_image_token_in_minigpt4(
                    input_ids,
                    language_model_inputs,
                    attention_mask,
                    image_token_indexes,
                    pixel_values,
                    batch_size)
            input_ids = None
        else:
            # do the processing in the vision model
            # language is the causallm model.
            # so use language model.model to do the embed_tokens
            if pixel_values.dim() == 3:
                # the batch dim is missing;
                pixel_values = pixel_values[None]
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
                self.vision_model.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask,
                    None, None,
                    pixel_values,
                    self.language_projection,
                    self.language_model.model)
        # convert the dtype.
        # FIXME check when need to do this
        inputs_embeds = inputs_embeds.to(
            device=self.language_model.lm_head.weight.device)
        inputs_embeds = inputs_embeds.to(
            self.language_model.lm_head.weight.dtype)
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs
