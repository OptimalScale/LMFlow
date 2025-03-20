#!/usr/bin/env python
# coding=utf-8
"""This is a class called HFDecoderModel which is a wrapper around transformers model and
tokenizer classes. It has several methods such as __init__, tokenize, and train that are
used for training and fine-tuning the model. The __init__ method takes in several arguments
such as model_args, tune_strategy, and ds_config, which are used to load the pretrained
model and tokenizer, and initialize the training settings.

The tokenize method is used to tokenize the input text and return the input IDs and attention
masks that can be fed to the model for training or inference.

This class supports different tune_strategy options such as 'normal', 'none', 'lora', and
'adapter', which allow for different fine-tuning settings of the model. However, the 'lora'
and 'adapter' strategies are not yet implemented.

Overall, this class provides a convenient interface for loading and fine-tuning transformer
models and can be used for various NLP tasks such as language modeling, text classification,
and question answering.
"""

import copy
import logging
import time
from typing import List, Union

import deepspeed
import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
)
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoModel,
    AutoProcessor,
    LlamaConfig
)
from transformers.testing_utils import CaptureLogger

from lmflow.datasets.dataset import Dataset
from lmflow.models.encoder_decoder_model import EncoderDecoderModel
from lmflow.models.interfaces.tunable import Tunable
from lmflow.models.vision2seq_model import CustomAutoVision2SeqModel
from lmflow.utils.multimodal import update_custom_config, load_llava_pretrain_model
from lmflow.utils.versioning import is_package_version_at_least

if is_package_version_at_least("transformers", "4.46.0"):
    from transformers.integrations.deepspeed import HfDeepSpeedConfig, HfTrainerDeepSpeedConfig
else:
    from transformers.deepspeed import HfDeepSpeedConfig, HfTrainerDeepSpeedConfig


logger = logging.getLogger(__name__)


class HFEncoderDecoderModel(EncoderDecoderModel, Tunable):
    r"""
    Initializes a HFEncoderDecoderModel instance.

    Parameters
    ------------

    model_args :
        Model arguments such as model name, path, revision, etc.

    tune_strategy : str or none,  default="normal".
        A string representing the dataset backend. Defaults to "huggingface".

    ds_config :
        Deepspeed configuations.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.
    """

    def __init__(
        self,
        model_args,
        tune_strategy='normal',
        ds_config=None,
        device="gpu",
        use_accelerator=False,
        custom_model=False,
        with_deepspeed=True,
        pipeline_args=None,
        *args,
        **kwargs
    ):
        """
        Initializes a HFDecoderModel instance.
        :param model_args: dictionary with model arguments such as model name, path, revision, etc.
        :param tune_strategy: tuning strategy: normal, none, lora or adapter
        :param ds_config: deepspeed configuration for distributed training
        """

        # See more about loading any type of standard or custom dataset (from
        # files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer
        #
        # Distributed training: The .from_pretrained methods guarantee that
        # only one local process can concurrently download model & vocab.

        self.device = device

        if tune_strategy == 'normal':
            raise NotImplementedError(
                f"tune_strategy \"{tune_strategy}\" is not supported"
            )
        elif tune_strategy == 'none':
            if use_accelerator:
                raise NotImplementedError(
                    f"Currently encoder2decoder model is not supported with accelerator"
                )
            # dschf = HfDeepSpeedConfig(ds_config)
            dschf = HfTrainerDeepSpeedConfig(ds_config)
            if pipeline_args is not None:
                dschf.trainer_config_process(pipeline_args)
            peft_model_id = model_args.lora_model_path
            # NOTE: Currently offload is not supported by llama
            if "llama" in model_args.model_name_or_path and model_args.use_ram_optimized_load:
                logger.warning(
                    "llama does not support RAM optimized load. Automatically"
                    " use original load instead."
                )
                model_args.use_ram_optimized_load = False

            # get model register
            self.arch_type = model_args.arch_type
            if self.arch_type == "encoder_decoder":
                if model_args.model_name_or_path == 'THUDM/chatglm-6b':
                    model_register = AutoModel
                else:
                    model_register = AutoModelForSeq2SeqLM
            elif self.arch_type == "vision_encoder_decoder":
                if not custom_model:
                    model_register = AutoModelForVision2Seq
                else:
                    model_register = CustomAutoVision2SeqModel
            else:
                raise NotImplementedError
            if not custom_model:
                if model_args.model_name_or_path == 'THUDM/chatglm-6b':
                    self.backend_model = model_register.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

                elif model_args.use_ram_optimized_load and peft_model_id is None:
                    try:
                        # RAM-optimized load
                        self.backend_model = model_register.from_pretrained(
                            model_args.model_name_or_path,
                            device_map="auto",
                            offload_folder="offload",
                            offload_state_dict=True,
                        )
                    except:
                        logger.warning(
                            "Failed to use RAM optimized load. Automatically"
                            " use original load instead."
                        )
                        # Normal load
                        self.backend_model = model_register.from_pretrained(
                            model_args.model_name_or_path,
                        )
                else:
                    if peft_model_id is not None:
                        logger.warning(
                            "LoRA does not support RAM optimized load currently."
                            " Automatically use original load instead."
                        )
                    self.backend_model = model_register.from_pretrained(
                        model_args.model_name_or_path,
                    )
            # else:
            #     self.backend_model = model_register.from_pretrained(
            #         model_args.model_name_or_path)
            else:
                if model_args.llava_loading is False:
                    # FIXME remove the following from_pretrained code by
                    # creating a unified pretrained model.
                    model = CustomAutoVision2SeqModel.from_pretrained(model_args.model_name_or_path)
                    if model_args.llm_model_name_or_path is not None:
                        text_config = LlamaConfig.from_pretrained(model_args.llm_model_name_or_path)
                        model.config.text_config = text_config
                    model.language_model_from_pretrained(model_args.llm_model_name_or_path,
                                                        low_resource=model_args.low_resource)
                    state_dict = torch.load(
                        model_args.pretrained_language_projection_path,
                        map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                else:
                    config = AutoConfig.from_pretrained(
                        model_args.model_name_or_path)
                    if model_args.low_resource:
                        kwargs = dict(
                            torch_dtype=torch.float16,
                            load_in_8bit=True,
                            device_map="auto",
                        )
                    else:
                        # kwargs = dict(torch_dtype=torch.float16)
                        kwargs = dict(device_map="auto")
                    if (model_args.image_encoder_name_or_path is None and
                        model_args.qformer_name_or_path is None and
                        model_args.llm_model_name_or_path is None):
                        config = AutoConfig.from_pretrained(
                            model_args.model_name_or_path)
                        model = CustomAutoVision2SeqModel.from_pretrained(
                            model_args.model_name_or_path, **kwargs)
                    else:
                        config = update_custom_config(config, model_args)
                        model = CustomAutoVision2SeqModel(
                            config,
                            image_encoder_name_or_path=model_args.image_encoder_name_or_path,
                            qformer_name_or_path=model_args.qformer_name_or_path,
                            language_model_name_or_path=model_args.llm_model_name_or_path,
                            low_resource=model_args.low_resource)
                        if model_args.pretrained_language_projection_path is not None:
                            state_dict = torch.load(
                                model_args.pretrained_language_projection_path, map_location="cpu")
                            new_state_dict = {}
                            new_state_dict['model.language_projection.weight'] = \
                                state_dict['model.mm_projector.weight']
                            new_state_dict['model.language_projection.bias'] = \
                                state_dict['model.mm_projector.bias']
                if model_args.llava_pretrain_model_path is not None:
                    # used for inference that directly load the preatrain model
                    model = load_llava_pretrain_model(
                        model, model_args.llava_pretrain_model_path)
                    if model_args.save_pretrain_model_path is not None:
                        model.save_pretrained(
                            model_args.save_pretrain_model_path)
                self.backend_model = model
            # init tokenizer
            if self.arch_type == "encoder_decoder":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path, trust_remote_code=True)
            elif self.arch_type == "vision_encoder_decoder":
                if model_args.llava_loading is False:
                    # blip2 image and token processor
                    self.tokenizer = AutoProcessor.from_pretrained(
                        model_args.model_name_or_path, trust_remote_code=True)
                    if model_args.llm_model_name_or_path is not None:
                        # update the tokenizer from the custom llm.
                        self.tokenizer.tokenizer = (
                            AutoTokenizer.from_pretrained(
                                model_args.llm_model_name_or_path)
                        )
                    self.image_processor = self.tokenizer.image_processor

                else:
                    # image processor is stored in the vision encoder
                    if model_args.llm_model_name_or_path is not None:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_args.llm_model_name_or_path)
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            config.text_config._name_or_path)
                    self.image_processor = self.backend_model.image_processor
            else:
                raise NotImplementedError

            self.backend_model_full = self.backend_model
            if peft_model_id is not None:
                self.backend_model = PeftModel.from_pretrained(
                    self.backend_model, peft_model_id
                )
            if tune_strategy == "none" and with_deepspeed is True:
                # when load the model with 4bit / 8bit.
                # fail to use deepspeed.
                if device == "gpu":
                    deepspeed.init_distributed()
                    self.ds_engine = deepspeed.initialize(model=self.backend_model, config_params=ds_config)[0]
                    self.ds_engine.module.eval()

            self.tokenizer.padding_side = "left" # necessary for auto-gressive inference

        elif tune_strategy == 'adapter':
            raise NotImplementedError('adapter tune strategy not implemented')

        if self.arch_type == "encoder_decoder":
            if self.tokenizer.eos_token_id is None:
                self.tokenizer.eos_token_id = self.backend_model.config.eos_token_id
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenize(self, dataset, *args, **kwargs):
        """
        Tokenize the full dataset.

        Parameters
        ------------
        dataset :
            Text dataset.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset.
        """
        raise NotImplementedError('tokenize not implemented')

    def encode(self, input: Union[str, List[str]], *args, **kwargs ) -> Union[List[int], List[List[int]]]:
        """
        Perform encoding process of the tokenizer.

        Parameters
        ------------
        inputs : str or list.
            The text sequence.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The tokenized inputs.
        """
        # check how to handle the image processor
        if isinstance(input, dict):
            # TODO refactor the input type to make it elegant.
            kwargs.update(input)
            if "images" not in input:
                tokens = self.tokenizer(*args, **kwargs)
            else:
                if getattr(self.tokenizer, "image_processor", None) is not None:
                    tokens = self.tokenizer(*args, **kwargs)
                elif getattr(self, "image_processor", None) is not None:
                    images = kwargs.pop("images")
                    tokens = self.tokenizer(*args, **kwargs)
                    images = self.image_processor.preprocess(
                        images, return_tensors='pt')['pixel_values'][0]
                    tokens['pixel_values'] = images
                else:
                    print("Can not find the image processor")
                    raise NotImplementedError
            return tokens
        elif isinstance(input, list):
            return self.tokenizer(text=input, *args, **kwargs)#batch encode,will automatically do left padding
        elif isinstance(input, str):
            return self.tokenizer.encode(text=input, *args, **kwargs)
        else:
            raise NotImplementedError(f'type "{type(input)}" cannot be encoded')


    def decode(self, input, *args, **kwargs ) -> Union[str, List[str]]:
        """
        Perform decoding process of the tokenizer.

        Parameters
        ------------
        inputs : list.
            The token sequence.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The text decoded from the token inputs.
        """
        if isinstance(input, List):
            input=torch.tensor(input)
        if input.dim()==2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)#batch_decode
        else:
            # Can be list of ints or a Tensor
            return self.tokenizer.decode(input, *args, **kwargs)


    def inference(self, inputs, *args, **kwargs):
        """
        Perform generation process of the model.

        Parameters
        ------------
        inputs :
            The sequence used as a prompt for the generation or as model inputs to the model.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The generated sequence output
        """
        # current_time = time.strftime("%H:%M:%S", time.localtime())
        # print(f"{current_time}: model.inference: start", flush=True)

        # TODO need to discuss how to handle pad_token_id
        if self.arch_type == "encoder_decoder":
            kwargs.update(pad_token_id=self.tokenizer.pad_token_id)
        elif self.arch_type == "vision_encoder_decoder":
            # TODO disucss how to modify the interface to remove this part.
            inputs = copy.deepcopy(inputs)
            input_ids = inputs.pop('input_ids')
            kwargs.update(**inputs)
            inputs = input_ids

        # current_time = time.strftime("%H:%M:%S", time.localtime())
        # print(f"{current_time}: model.inference: kwargs update end", flush=True)

        with torch.no_grad():
            if self.device == "gpu":
                if getattr(self, "ds_engine", None) is not None:
                    outputs = self.ds_engine.module.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        *args,
                        **kwargs
                    )
                else:
                    outputs = self.backend_model.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        *args,
                        **kwargs,
                    )
            elif self.device == "cpu":
                outputs = self.backend_model.generate(
                    input_ids=inputs,
                    synced_gpus=True,
                    *args,
                    **kwargs
                )
            else:
                raise NotImplementedError(
                    f"device \"{self.device}\" is not supported"
                )

        # current_time = time.strftime("%H:%M:%S", time.localtime())
        # print(f"{current_time}: model.inference: end", flush=True)

        return outputs


    def merge_lora_weights(self):
        if self.model_args.use_lora:
            self.get_backend_model().merge_and_unload()
        else:
            logger.warning("LoRA training is NOT enabled. Merging LoRA weights is not applicable.")


    def save(self, dir, save_full_model=False, *args, **kwargs):
        """
        Perform generation process of the model.

        Parameters
        ------------
        dir :
            The directory to save model and tokenizer

        save_full_model : Optional.
            Whether to save full model.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The generated sequence output
        """
        self.get_tokenizer().save_pretrained(dir)
        if save_full_model and self.model_args.use_lora:
            self.backend_model_full.save_pretrained(dir)
        else:
            self.get_backend_model().save_pretrained(dir)


    def get_max_length(self):
        """
        Return max acceptable input length in terms of tokens.
        """
        if "tokenizer" not in self.tokenizer.__dict__:
            return self.tokenizer.model_max_length
        else:
            # for the multi-modality processor,
            # the max length is stored in the inner text tokenizer
            return self.tokenizer.tokenizer.model_max_length


    def get_tokenizer(self):
        """
        Return the tokenizer of the model.
        """
        return self.tokenizer


    def get_backend_model(self):
        """
        Return the backend model.
        """
        return self.backend_model
