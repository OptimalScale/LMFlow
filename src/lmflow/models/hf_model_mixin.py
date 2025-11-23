#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import copy
import gc
import logging
from contextlib import nullcontext
from typing import Literal, Optional, Union

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.modeling_utils import is_fsdp_enabled

from lmflow.args import ModelArguments
from lmflow.models.base_model import BaseModel
from lmflow.utils.constants import LMFLOW_LORA_TARGET_MODULES_MAPPING
from lmflow.utils.envs import is_accelerate_env
from lmflow.utils.versioning import is_deepspeed_available, is_vllm_available, is_sglang_available


logger = logging.getLogger(__name__)


HF_AUTOMODEL_MAPPING = {"decoder_only": AutoModelForCausalLM, "text_regression": AutoModelForSequenceClassification}

HF_AUTOMODEL_TYPE = Union[AutoModelForCausalLM, AutoModelForSequenceClassification]

LORA_TARGET_MODULES_MAPPING = {
    k: TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(k, LMFLOW_LORA_TARGET_MODULES_MAPPING.get(k))
    for k in set(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING) | set(LMFLOW_LORA_TARGET_MODULES_MAPPING)
}


class HFModelMixin(BaseModel):
    def __init__(
        self,
        model_args: ModelArguments,
        do_train: bool,
        device: Optional[str] = "gpu",
        hf_auto_model_additional_args: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """Initializes a HFModel instance.

        Parameters
        ----------
        model_args :
            Dictionary with model arguments such as model name, path, revision, etc.
        do_train : bool
            To prepare the model for training or inference.
        device : str, optional
            By default "gpu"
        """

        # See more about loading any type of standard or custom dataset (from
        # files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer
        #
        # Distributed training: The .from_pretrained methods guarantee that
        # only one local process can concurrently download model & vocab.

        self.device = device
        self.model_args = model_args
        self.hf_auto_model = HF_AUTOMODEL_MAPPING[model_args.arch_type]
        self.do_train = do_train

        self.tokenizer = self.__prepare_tokenizer(model_args)
        self.torch_dtype = self.__prepare_dtype(model_args)
        self.hf_model_config = self.__prepare_model_config(model_args, hf_auto_model_additional_args)
        self.quant_config = self.__prepare_quant_config(model_args)
        self.peft_config = self.__prepare_peft_config(model_args)
        self._activated = False  # for inference load and offload

        # Some implementations require custom modules to be injected into the model.
        self.__model_module_inject(model_args)

        if self.do_train:
            self.__prepare_model_for_training(model_args, self.hf_auto_model)

        self.__fix_special_tokens()

    def __prepare_tokenizer(
        self,
        model_args: ModelArguments,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
        if not tokenizer_name:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is"
                " not supported by this script. You can do it from another"
                " script, save it, and load it from here, using"
                " --tokenizer_name."
            )

        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "token": model_args.token,
            "trust_remote_code": model_args.trust_remote_code,
        }
        if model_args.padding_side != "auto":
            tokenizer_kwargs["padding_side"] = model_args.padding_side

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        except RecursionError:
            logger.warning(
                "The tokenizer_config.json file doesn't set the special tokens. Using default values: "
                "<unk>, <s>, </s> for unknown token, bos token and eos token respectively."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, unk_token="<unk>", bos_token="<s>", eos_token="</s>", **tokenizer_kwargs
            )

        tokenizer.truncation_side = model_args.truncation_side or tokenizer.truncation_side
        tokenizer.model_max_length = model_args.model_max_length or tokenizer.model_max_length

        return tokenizer

    def __prepare_dtype(
        self,
        model_args: ModelArguments,
    ) -> torch.dtype:
        if model_args.arch_type == "text_regression":
            if model_args.torch_dtype in ["auto", None, "bf16", "bfloat16"]:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = getattr(torch, model_args.torch_dtype)
                logger.warning(
                    f"If you are doing reward modeling,"
                    f" InstructGPT uses torch.bfloat16 for reward model, but you"
                    f" are using {torch_dtype} for your reward model init. Ignore"
                    f" this warning if it is intended."
                )
        else:
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )

        logger.debug(f"torch_dtype on init: {torch_dtype}")

        return torch_dtype

    def __prepare_model_config(
        self,
        model_args: ModelArguments,
        hf_auto_model_additional_args: Optional[dict] = None,
    ):
        """Prepare model configuration for hf auto register,
        Parameters
        ----------
        model_args : ModelArguments
            LMFlow model arguments.
        hf_auto_model_additional_args : Optional[dict], optional
            Special configurations such as `num_labels` in `AutoModelForSequenceClassification`
            (commonly used in reward modeling) will not preset in __prepare_model_config,
            so it should be passed in hf_auto_model_additional_args.
        Returns
        -------
        config : ModelConfig
            hf model config.
        """
        config_kwargs = {
            "attn_implementation": "flash_attention_2" if model_args.use_flash_attention else None,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.token,
            "trust_remote_code": model_args.trust_remote_code,
            "from_tf": bool(".ckpt" in model_args.model_name_or_path),
        }
        if hf_auto_model_additional_args is not None:
            config_kwargs.update(hf_auto_model_additional_args)

        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")

        return config

    def __prepare_quant_config(
        self,
        model_args: ModelArguments,
    ):
        quant_config = None
        if self.do_train:
            if model_args.use_qlora:
                if model_args.quant_bit == 8:
                    if is_fsdp_enabled():
                        raise ValueError("FSDP + Qlora 8-bit quantization is not supported.")
                    quant_config_kwargs = {
                        "load_in_8bit": True,
                    }
                elif model_args.quant_bit == 4:
                    logger.warning(
                        "For users who are using Accelerate (FSDP backend) or DeepSpeed, "
                        "we only implement Qlora 4-bit quantization with torch.bfloat16 dtype currently. "
                        "Carefully check the Accelerate or DeepSpeed configurations, since they may cast dtype "
                        "and cause errors like "
                        "(DeepSpeed) `TypeError: output tensor must have the same type as input tensor`, or "
                        "(Accelerate FSDP) `ValueError: Must flatten tensors with uniform dtype but got "
                        "torch.bfloat16 and torch.float32`. "
                        "Consider using other peft methods if your device doesn't support torch.bfloat16. "
                        "(This is just a notification and please self-check the compatibility of your device.)"
                    )
                    quant_config_kwargs = {
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.bfloat16,
                        "bnb_4bit_use_double_quant": model_args.double_quant,
                        "bnb_4bit_quant_type": model_args.quant_type,
                        "bnb_4bit_quant_storage": torch.bfloat16,  # fsdp+qlora, see https://huggingface.co/docs/bitsandbytes/v0.43.3/en/fsdp_qlora
                    }
                else:
                    raise ValueError("Qlora only supports 4-bit and 8-bit.")

                quant_config = BitsAndBytesConfig(**quant_config_kwargs)

        else:  # inference
            if model_args.use_int8:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=model_args.use_int8,
                )

        return quant_config

    def __prepare_peft_config(
        self,
        model_args: ModelArguments,
    ):
        peft_config = None
        if model_args.use_lora:
            if model_args.lora_target_modules:
                lora_target_modules = model_args.lora_target_modules
            else:
                model_config = self.hf_model_config
                if hasattr(model_config, "to_dict"):
                    model_config = model_config.to_dict()
                if "model_type" not in model_config or not model_config["model_type"]:
                    logger.warning(
                        "It seems that your base model is a custom model, since "
                        "model_type is not found in model_config when preparing peft config. "
                        "Setting model_type to 'custom' as a fallback."
                    )
                    model_config["model_type"] = "custom"
                lora_target_modules = LORA_TARGET_MODULES_MAPPING.get(model_config["model_type"], None)

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=lora_target_modules,
            )
        if model_args.use_dora:
            if model_args.lora_target_modules:
                lora_target_modules = model_args.lora_target_modules
            else:
                model_config = self.hf_model_config
                if hasattr(model_config, "to_dict"):
                    model_config = model_config.to_dict()
                if "model_type" not in model_config or not model_config["model_type"]:
                    logger.warning(
                        "It seems that your base model is a custom model, since "
                        "model_type is not found in model_config when preparing peft config. "
                        "Setting model_type to 'custom' as a fallback."
                    )
                    model_config["model_type"] = "custom"
                lora_target_modules = LORA_TARGET_MODULES_MAPPING.get(model_config["model_type"], None)

            peft_config = LoraConfig(
                use_dora=True,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=lora_target_modules,
            )
        return peft_config

    def __model_module_inject(
        self,
        model_args: ModelArguments,
    ) -> None:
        """Override some model modules with custom implementations.

        Current implementations:
        - Position interpolation (model_args.do_rope_scaling):
            replace llama embeddings with condense embeddings.
        """
        # position interpolation
        if model_args.do_rope_scaling:
            if "LlamaForCausalLM" in self.model_config.architectures:
                from lmflow.utils.position_interpolation.llama_rope_scaled_monkey_patch import (
                    replace_llama_with_condense,
                )

                replace_llama_with_condense(model_args.rope_pi_ratio, model_args.rope_ntk_ratio)

    def __prepare_model_for_training(
        self,
        model_args: ModelArguments,
        hf_auto_model: HF_AUTOMODEL_TYPE,
    ):
        assert self.do_train, "To prepare the model for training, please set do_train=True."
        # TODO: change to accelerate

        if "hymba" in model_args.model_name_or_path:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True

        logger.info("Preparing model for training")
        if model_args.model_name_or_path:
            model = hf_auto_model.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=self.torch_dtype,
                config=self.hf_model_config,
                quantization_config=self.quant_config,
                trust_remote_code=model_args.trust_remote_code,
            )

        else:
            model = hf_auto_model.from_config(self.hf_model_config)
            n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
            logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")
        self.backend_model_full = model

        if model_args.ignore_bias_buffers:
            # torch distributed hack
            # fix for DDP issues with LM bias/mask buffers - invalid scalar type, inplace operation.
            # See: https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992
            model._ddp_params_and_buffers_to_ignore = [
                name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
            ]

        if model_args.use_lora or model_args.use_dora:
            model.enable_input_require_grads()
            model = get_peft_model(model, self.peft_config)
            model.print_trainable_parameters()

        # We resize the embeddings only when necessary to avoid index errors.
        # If you are creating a model from scratch on a small vocab and want a
        # smaller embedding size, remove this test.
        resize_embedding_context = nullcontext()
        if is_deepspeed_available() and not is_accelerate_env():
            import deepspeed

            resize_embedding_context = deepspeed.zero.GatheredParameters(
                model.get_input_embeddings().weight, modifier_rank=None
            )

        with resize_embedding_context:
            weights = model.get_input_embeddings().weight
            embedding_size = weights.shape[0]

        if len(self.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(self.tokenizer))

        self.backend_model = model

    def __prepare_model_for_inference(
        self,
        model_args: ModelArguments,
        hf_auto_model: HF_AUTOMODEL_TYPE,
    ):
        logger.info(f"Backend model already initialized, moving to device: {self.device}")
        if hasattr(self, "backend_model"):
            if self.backend_model.device == torch.device("cpu"):
                self.backend_model.to(self.device)
            return

        # TODO: change to accelerate
        logger.info("Preparing model for inference")
        inference_load_kwargs = {}
        inference_load_kwargs_bak = copy.deepcopy(inference_load_kwargs)
        ram_optimized_load_kwargs = {
            "device_map": "auto",
            "offload_folder": "offload",
            "offload_state_dict": True,
        }

        if model_args.use_ram_optimized_load:
            inference_load_kwargs.update(ram_optimized_load_kwargs)

        try:
            self.backend_model = hf_auto_model.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=self.torch_dtype,
                config=self.hf_model_config,
                quantization_config=self.quant_config,
                **inference_load_kwargs,
            )
        except Exception:
            logger.warning("Failed to use RAM optimized load. Using original load instead.")
            self.backend_model = hf_auto_model.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=self.torch_dtype,
                config=self.hf_model_config,
                quantization_config=self.quant_config,
                **inference_load_kwargs_bak,
            )

        self.backend_model_full = self.backend_model

        if model_args.lora_model_path is not None:
            self.backend_model = PeftModel.from_pretrained(
                self.backend_model,
                model_args.lora_model_path,
            )

        if self.device == "gpu" and not is_accelerate_env():
            if is_deepspeed_available():
                import deepspeed

                deepspeed.init_distributed()
                self.ds_engine = deepspeed.initialize(model=self.backend_model)[0]
                self.ds_engine.module.eval()
            else:
                raise ImportError("Deepspeed is not available. Please install via `pip install -e '.[deepspeed]'`.")

    def __prepare_model_for_vllm_inference(
        self,
        model_args: ModelArguments,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
    ):
        if not is_vllm_available():
            raise ImportError('VLLM is not available. Please install via `pip install -e ".[vllm]"`.')
        
        from vllm import LLM

        self.backend_model_for_inference = LLM(
            model=model_args.model_name_or_path,
            tokenizer=model_args.model_name_or_path,
            dtype=model_args.torch_dtype if model_args.torch_dtype else "auto",
            load_format="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        
    def __prepare_model_for_sglang_inference(
        self,
        model_args: ModelArguments,
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: Optional[int] = None,
    ):
        if not is_sglang_available():
            raise ImportError('SGLang is not available. Please install via `pip install -e ".[sglang]"`.')
        
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.server_args import ServerArgs
        
        sgl_server_args = ServerArgs(
            model_path=model_args.model_name_or_path,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
        )
        self.backend_model_for_inference = Engine(server_args=sgl_server_args)

    def __fix_special_tokens(self):
        # old models/tokenizers may not have these attributes, fixing
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.hf_model_config.eos_token
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.hf_model_config.eos_token_id

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.model_args.eos_padding:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not hasattr(self.hf_model_config, "pad_token_id"):
            logger.warning("pad_token_id not found in model config. Setting pad_token_id to eos_token_id.")
            self.hf_model_config.pad_token_id = self.hf_model_config.eos_token_id
        elif self.hf_model_config.pad_token_id is None:
            logger.warning("pad_token_id is None in model config. Setting pad_token_id to eos_token_id.")
            self.hf_model_config.pad_token_id = self.hf_model_config.eos_token_id

    def activate_model_for_inference(
        self,
        inference_engine: Literal["huggingface", "vllm", "sglang"] = "huggingface",
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: Optional[int] = None,
    ):
        if self._activated:
            logger.warning("You are trying to activate the model for inference, but it is already activated.")
            return

        if inference_engine == "vllm":
            self.__prepare_model_for_vllm_inference(
                model_args=self.model_args,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif inference_engine == "sglang":
            self.__prepare_model_for_sglang_inference(
                model_args=self.model_args,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            self.__prepare_model_for_inference(
                model_args=self.model_args,
                hf_auto_model=self.hf_auto_model,
            )

        self._activated = True

    def deactivate_model_for_inference(
        self,
        inference_engine: Literal["huggingface", "vllm", "sglang"] = "huggingface",
    ):
        """Deactivate the model and release the resources.

        NOTE: Currently, VLLM doesn't have an official way to do this, and the
        implementation below cannot release all gpu resources by our observation.
        Thus this method is just a placeholder for future implementation. See:
        [Github issue](https://github.com/vllm-project/vllm/issues/1908)
        """
        if not self._activated:
            logger.warning("You are trying to deactivate the model for inference, but it is already deactivated.")
            return

        if inference_engine == "vllm":
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            del self.backend_model_for_inference.llm_engine.model_executor.driver_worker
            del self.backend_model_for_inference
            gc.collect()
            torch.cuda.empty_cache()
        elif inference_engine == "sglang":
            pass
        else:
            self.backend_model.to("cpu")

        self._activated = False

    def get_max_length(self):
        """
        Return max acceptable input length in terms of tokens.
        """
        return self.tokenizer.model_max_length

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
