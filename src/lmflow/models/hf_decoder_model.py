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

import hashlib
import logging
import os, shutil
from typing import List, Union, Optional, Dict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import PeftModel

from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_model_mixin import HFModelMixin
from lmflow.models.decoder_model import DecoderModel
from lmflow.models.interfaces.tunable import Tunable
from lmflow.utils.constants import (
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
    CONVERSATION_DATASET_DESCRIPTION,
)
from lmflow.utils.conversation_template import PRESET_TEMPLATES
from lmflow.utils.data_utils import VLLMInferenceResultWithInput
from lmflow.tokenization.hf_decoder_model import (
    tokenize_function, 
    conversation_tokenize_function
)
from lmflow.utils.versioning import is_ray_available, is_vllm_available, is_flash_attn_available


logger = logging.getLogger(__name__)


if is_flash_attn_available():
    import flash_attn
else:
    logger.warning("Consider install flash_attn for better performance.")
    
if is_vllm_available():
    from vllm import SamplingParams
    
if is_ray_available():
    import ray
    import ray.data


class HFDecoderModel(DecoderModel, HFModelMixin, Tunable):
    r"""
    Initializes a HFDecoderModel instance.

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
        *args,
        **kwargs
    ):
        """
        Initializes a HFDecoderModel instance.
        :param model_args: dictionary with model arguments such as model name, path, revision, etc.
        :param tune_strategy: tuning strategy: normal, none, lora or adapter
        :param ds_config: deepspeed configuration for distributed training
        """
        HFModelMixin.__init__(
            self,
            model_args=model_args,
            do_train=True if tune_strategy == "normal" else False,
            ds_config=ds_config,
            device=device,
            use_accelerator=use_accelerator,
            *args,
            **kwargs
        )


    def tokenize(
        self, 
        dataset, 
        add_special_tokens=True, 
        *args, 
        **kwargs
    ) -> Dataset:
        """
        Tokenize the full dataset.
    
        Parameters
        ------------
        dataset : lmflow.datasets.Dataset.

        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset, without any leading or trailing special
            tokens (normally they are Begin-Of-Sentence or End-Of-Sentence
            tokens).
        """
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        dataset_type = dataset.get_type()
        model_args = self.model_args
        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features)
        data_args = raw_datasets.get_data_args()

        # Requires three types of information for tokenizing different datasets
        #   1) Which fields require tokenization, e.g.
        #        "text2float": "text", but not "float"
        #        "text2text": both "input" and "output"
        #   2) How will there tokenized sequence concatenated together, e.g.
        #        "text_only": "text" -> "text"
        #        "text2text": "input", "output" -> "input" + "output"
        #   3) Which fields require loss in final computation, e.g.
        #        "text_only": "text"
        #        "text2text": "output" only
        tokenized_column_order = None       # Handles 1) and 2)
        label_columns = None                # Handles 3)
        if dataset_type == "text_only":
            tokenized_column_order = ["text"]
            label_columns = ["text"]
        elif dataset_type == "text2text":
            tokenized_column_order = ["input", "output"]
            label_columns = ["output"]
            add_special_tokens = False
        elif dataset_type == "conversation":
            if data_args.conversation_template:
                if data_args.conversation_template in PRESET_TEMPLATES.keys():
                    conversation_template = PRESET_TEMPLATES[data_args.conversation_template]
                else:
                    raise NotImplementedError(
                        f"Conversation template {data_args.conversation_template} is not supported yet."
                    )
            else:
                logger.warning("No conversation template provided. Using default template.")
                conversation_template = PRESET_TEMPLATES['empty']
                        
            logger.warning(f"Conversation template: {conversation_template}")
        else:
            raise NotImplementedError(
                f"dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types:\n"
                f"    1) {TEXT_ONLY_DATASET_DESCRIPTION}\n"
                f"    2) {TEXT2TEXT_DATASET_DESCRIPTION}\n"
                f"    3) {CONVERSATION_DATASET_DESCRIPTION}\n"
            )

        # Whether to truncate long sequences to fit into max_length
        use_truncation = False
        if model_args.use_lora or data_args.disable_group_texts:
            use_truncation = True
        
        tokenize_fn = conversation_tokenize_function if "conversation" in dataset_type else tokenize_function
        tokenize_fn_kwargs = {
            "data_args": data_args,
            "tokenizer": self.tokenizer,
            "column_names": column_names,
        }
        if "conversation" in dataset_type:
            tokenize_fn_kwargs["conversation_template"] = conversation_template
        else:
            tokenize_fn_kwargs["label_columns"] = label_columns
            tokenize_fn_kwargs["tokenized_column_order"] = tokenized_column_order
            tokenize_fn_kwargs["add_special_tokens"] = add_special_tokens
            tokenize_fn_kwargs["use_truncation"] = use_truncation
                           
        tokenize_kwargs = {}
        if not data_args.streaming:
            fingerprint = hashlib.md5(
                (
                    raw_datasets.get_fingerprint()
                    + str(self.tokenizer)
                    + f'###padding_side={self.tokenizer.padding_side}'
                    + ('###conversation_template=' + str(conversation_template) if "conversation" in dataset_type else "")
                    + f'###disable_group_texts={data_args.disable_group_texts}'
                    + f'###block_size={data_args.block_size}'
                ).encode("utf-8")
            ).hexdigest()
            tokenize_kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                "load_from_cache_file": not data_args.overwrite_cache,
                "desc": "Running tokenizer on dataset",
                "new_fingerprint": fingerprint,
            }

        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=column_names,
            fn_kwargs=tokenize_fn_kwargs,
            **tokenize_kwargs
        )

        return tokenized_datasets


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
            if string input,return the tokenized inputs.
            "Hello,world!"-> [101, 7592, 1010, 2088, 102]
            if batch input,return {input_ids,attention_mask,token_type_ids}
            ["Hello,world!","Hello!"]-> {'input_ids': tensor([[  101,  7592,  1010,  2088,   102],...),'attention_mask': tensor([[1, 1, 1, 1, 1],[0,0,1,1,1]])}
        """
        if isinstance(input, list):
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
        inputs : list or tensor.
            The token sequence.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The text decoded from the token inputs.
            if batch input,return the list of text
            [[101, 7592, 1010, 2088, 102],[101, 7592, 1010, 2088, 102]]-> ["Hello,world!","Hello,world!"
            if single input,return the text
            [101, 7592, 1010, 2088, 102]-> "Hello,world!"
        """
        if isinstance(input, List):
            input=torch.tensor(input)
        if input.dim()==2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)#batch_decode
        else:
            # Can be list of ints or a Tensor
            return self.tokenizer.decode(input, *args, **kwargs)

        
    def inference(
        self, 
        inputs, 
        release_gpu: bool = False,
        use_vllm: bool = False,
        **kwargs
    ):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        inputs :
            The sequence used as a prompt for the generation or as model inputs to the model.
            When using vllm inference, this should be a string or a list of strings.
            When using normal inference, this should be a tensor.
        release_gpu : bool, optional
            Whether to release the GPU resource after inference, by default False.
        use_vllm : bool, optional
            Whether to use VLLM for inference, by default False.
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """
        if not self._activated:
            self.activate_model_for_inference(
                use_vllm=use_vllm,
                **kwargs,
            )
            
        if use_vllm:
            if not is_vllm_available():
                raise ImportError("vllm is not installed. Please install vllm to use VLLM inference.")
            res = self.__vllm_inference(inputs, **kwargs)
        else:
            res = self.__inference(inputs, **kwargs)
            
        if release_gpu:
            self.deactivate_model_for_inference(use_vllm=use_vllm)
            
        return res


    def __inference(self, inputs, *args, **kwargs):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        inputs :
            The **tokenized** sequence used as a prompt for the generation or as model inputs to the model.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """
        with torch.no_grad():
            if self.use_accelerator:
                outputs = self.backend_model.generate(
                    input_ids=inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    *args,
                    **kwargs
                )
            else:
                if self.device == "gpu":
                    outputs = self.ds_engine.module.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        *args,
                        **kwargs
                    )
                elif self.device == "cpu":
                    outputs = self.backend_model.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        *args,
                        **kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"device \"{self.device}\" is not supported"
                    )
        return outputs
    
    
    def __vllm_inference(
        self, 
        inputs: Union[str, List[str]],
        sampling_params: Optional['SamplingParams'] = None,
        **kwargs,
    ) -> List[VLLMInferenceResultWithInput]:
        """Perform VLLM inference process of the model.

        Parameters
        ----------
        inputs : Union[str, List[str]]
            Prompt(s), string or a list of strings.
        sampling_params : Optional[SamplingParams], optional
            vllm SamplingParams object, by default None.

        Returns
        -------
        List[VLLMInferenceResultWithInput]
            Return a list of VLLMInferenceResultWithInput, where each
            element contains the input prompt and the corresponding output.
            
            When `sampling_params.detokenize = True`, the output would be a list of strings,
            contains sampling_params.n samples for the corresponding prompt.
            
            When `sampling_params.detokenize = False`, return a list of list of ints 
            (token ids, no decoding after generation).
        """
        vllm_outputs = self.backend_model_for_inference.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        final_output = []
        for output in vllm_outputs:
            if sampling_params.detokenize:
                output_list = [sentence.text for sentence in output.outputs]  
            else:
                output_list = [sentence.token_ids for sentence in output.outputs]
                
            final_output.append({"input": output.prompt, "output": output_list})
                                
        return final_output
    
    
    def prepare_inputs_for_inference(
        self,
        dataset: Dataset,
        apply_chat_template: bool = True,
        enable_distributed_inference: bool = False,
        use_vllm: bool = False,
        **kwargs,
    ) -> Union[List[str], "ray.data.Dataset", Dict[str, torch.Tensor]]:
        """
        Prepare inputs for inference.
    
        Parameters
        ------------
        dataset : lmflow.datasets.Dataset.
            The dataset used for inference.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The prepared inputs for inference.
        """
        if use_vllm:
            if not is_ray_available() and enable_distributed_inference:
                raise ImportError("ray is not installed. Please install ray to use distributed vllm inference.")
            inference_inputs = self.__prepare_inputs_for_vllm_inference(
                dataset=dataset, 
                apply_chat_template=apply_chat_template,
                enable_distributed_inference=enable_distributed_inference,
            )
        else:
            inference_inputs = self.__prepare_inputs_for_inference(
                dataset,
                apply_chat_template=apply_chat_template,
                enable_distributed_inference=enable_distributed_inference,
            )
            
        return inference_inputs
    
    
    def __prepare_inputs_for_vllm_inference(
        self,
        dataset: Dataset,
        apply_chat_template: bool = True,
        enable_distributed_inference: bool = False,
    ) -> Union[List[str], "ray.data.Dataset"]:
        if dataset.get_type() == 'text_only':
            if apply_chat_template:
                dataset = dataset.map(
                    lambda sample: {
                        "templated": self.tokenizer.apply_chat_template(
                            [{"role":"user", "content": sample['text']}], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    },
                    num_proc=dataset.data_args.preprocessing_num_workers,
                )
                inference_inputs = dataset.get_backend_dataset()['templated']
            else:
                inference_inputs = dataset.get_backend_dataset()['text']
            
        elif dataset.get_type() == "text2text":
            logger.warning(f"For a text2text dataset, only `input` will be used as the model input.")
            if apply_chat_template:
                dataset = dataset.map(
                    lambda sample: {
                        "templated": self.tokenizer.apply_chat_template(
                            conversation=[{"role":"user", "content": sample['input']}], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    },
                    num_proc=dataset.data_args.preprocessing_num_workers,
                )
                inference_inputs = dataset.get_backend_dataset()['templated']
            else:
                inference_inputs = dataset.get_backend_dataset()['input']
            
        elif dataset.get_type() == 'conversation':
            if apply_chat_template:
                def preprocess_conversation(sample):
                    conversation = sample['messages'][:-1] if len(sample['messages'])%2 == 0 else sample['messages']
                        
                    if sample['messages'][-1]['role'] != 'user':
                        logger.warning(
                            "Not a valid conversation for generation, since the conversation "
                            "doesn't end up with an user message. Skip."
                        )
                        sample_out = {"templated": ""}
                    else:
                        sample_out = {"templated": self.tokenizer.apply_chat_template(
                            conversation=conversation,
                            tokenize=False,
                            add_generation_prompt=True,
                        )}
                        
                    return sample_out
                dataset = dataset.map(
                    preprocess_conversation,
                    num_proc=dataset.data_args.preprocessing_num_workers,
                )
                inference_inputs = dataset.get_backend_dataset()['templated']
            else:
                logger.warning(
                    "Your dataset is `conversation` type but `apply_chat_template` is set to False. "
                    "Will use the first user input in conversation as model input."
                )
                inference_inputs = [conversation[0]['content'] for conversation in dataset.get_backend_dataset()['messages']]

        else:
            raise NotImplementedError(
                f"Currently `{dataset.get_type()}` data are not supported for vllm inference."
            )

        inference_inputs = [sentence for sentence in inference_inputs if len(sentence) > 0]
        
        if enable_distributed_inference:
            inference_inputs = ray.data.from_items(inference_inputs) # -> Dict[str, np.ndarray], {"item": array(['...', '...', '...'])}
        
        return inference_inputs
    
    
    def __prepare_inputs_for_inference(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        raise NotImplementedError("prepare_inputs_for_inference is not implemented")


    def merge_lora_weights(self):
        if self.model_args.use_lora and not self.model_args.use_qlora:
            self.get_backend_model().merge_and_unload()
        elif self.model_args.use_qlora:
            logger.warning("Reloading base model in 16-bit precision to merge adapter weights. NOTE: Your device must have"
                           "sufficient memory to reload the model in half-precision without quantization.")
            self.get_peft_without_qlora()
            self.get_backend_model().merge_and_unload()
        else:
            logger.warning("LoRA training is NOT enabled. Merging LoRA weights is not applicable.")

    def get_peft_without_qlora(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)


            self.get_backend_model().save_pretrained(tmpdirname)

            torch_dtype = (
                self.model_args.torch_dtype
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            config_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "revision": self.model_args.model_revision,
                "token": self.model_args.token,
            }
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, **config_kwargs)
            device_map = "auto"
            if os.environ.get('LOCAL_RANK') is not None:
                local_rank = int(os.environ.get('LOCAL_RANK','0'))
                device_map = {'': local_rank}

            self.backend_model_full = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                token=self.model_args.token,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code = self.model_args.trust_remote_code,
                attn_implementation="flash_attention_2" if self.model_args.use_flash_attention else None,
            )
        
            self.backend_model = PeftModel.from_pretrained(self.backend_model_full, tmpdirname)

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
            save_dtype = (
                torch.float16
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            self.backend_model_full.to(dtype=save_dtype).save_pretrained(dir)
            logger.warning(f"Save full model with dtype: {save_dtype}")
        else:
            self.get_backend_model().save_pretrained(dir)
