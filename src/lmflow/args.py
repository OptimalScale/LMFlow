#!/usr/bin/env python
# coding=utf-8
"""This script defines dataclasses: ModelArguments and DatasetArguments,
that contain the arguments for the model and dataset used in training.

It imports several modules, including dataclasses, field from typing, Optional from typing,
require_version from transformers.utils.versions, MODEL_FOR_CAUSAL_LM_MAPPING,
and TrainingArguments from transformers.

MODEL_CONFIG_CLASSES is assigned a list of the model config classes from
MODEL_FOR_CAUSAL_LM_MAPPING. MODEL_TYPES is assigned a tuple of the model types
extracted from the MODEL_CONFIG_CLASSES.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers.utils.versions import require_version

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Define a class ModelArguments using the dataclass decorator. 
    The class contains several optional parameters that can be used to configure a model. 
    
    model_name_or_path : str
        a string representing the path or name of a pretrained
        model checkpoint for weights initialization. If None, a model will be trained from scratch.

    model_type :  str
        a string representing the type of model to use if training from
        scratch. If not provided, a pretrained model will be used.
    
    config_overrides :  str
        a string representing the default config settings to override
        when training a model from scratch.
    
    config_name : str
        a string representing the name or path of the pretrained config to
        use, if different from the model_name_or_path.
    
    tokenizer_name :  str
        a string representing the name or path of the pretrained tokenizer
        to use, if different from the model_name_or_path.

    cache_dir :  str
        a string representing the path to the directory where pretrained models
        downloaded from huggingface.co will be stored.

    use_fast_tokenizer : bool
        a boolean indicating whether to use a fast tokenizer (backed by the
        tokenizers library) or not.

    model_revision :  str
        a string representing the specific model version to use (can be a
        branch name, tag name, or commit id).

    use_auth_token : bool
        a boolean indicating whether to use the token generated when running
        huggingface-cli login (necessary to use this script with private models).

    torch_dtype :  str
        a string representing the dtype to load the model under. If auto is
        passed, the dtype will be automatically derived from the model's weights.

    use_ram_optimized_load : bool
        a boolean indicating whether to use disk mapping when memory is not
        enough.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    lora_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The incremental model diff introduced by LoRA finetuning."
                " Along with the original non-finetuned model forms the whole"
                " finetuned model."
            )
        }
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to lora."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "the rank of the lora parameters. The smaller lora_r is , the fewer parameters lora has."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Merging ratio between the fine-tuned model and the original. This is controlled by a parameter called alpha in the paper."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate in lora.linear."},
    )
    use_ram_optimized_load: bool = field(
        default=True,
        metadata={"help": "Whether use disk mapping when memory is not enough."}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DatasetArguments:
    """
    Define a class DatasetArguments using the dataclass decorator. 
    The class contains several optional parameters that can be used to configure a dataset for a language model. 
    

    dataset_path : str
        a string representing the path of the dataset to use.

    dataset_name : str
        a string representing the name of the dataset to use. The default value is "customized".

    is_custom_dataset : bool
        a boolean indicating whether to use custom data. The default value is False.

    customized_cache_dir : str
        a string representing the path to the directory where customized dataset caches will be stored.

    dataset_config_name : str
        a string representing the configuration name of the dataset to use (via the datasets library).

    train_file : str
        a string representing the path to the input training data file (a text file).

    validation_file : str
        a string representing the path to the input evaluation data file to evaluate the perplexity on (a text file).

    max_train_samples : int
        an integer indicating the maximum number of training examples to use for debugging or quicker training. 
        If set, the training dataset will be truncated to this number.

    max_eval_samples: int
        an integer indicating the maximum number of evaluation examples to use for debugging or quicker training. 
        If set, the evaluation dataset will be truncated to this number.

    streaming : bool
        a boolean indicating whether to enable streaming mode.

    block_size: int
        an integer indicating the optional input sequence length after tokenization. The training dataset will be 
        truncated in blocks of this size for training.

    The class also includes some additional parameters that can be used to configure the dataset further, such as `overwrite_cache`,
    `validation_split_percentage`, `preprocessing_num_workers`, `disable_group_texts`, `demo_example_in_prompt`, `explanation_in_prompt`,
    `keep_linebreaks`, and `prompt_structure`.

    The field function is used to set default values and provide help messages for each parameter. The Optional type hint is
    used to indicate that a parameter is optional. The metadata argument is used to provide additional information about 
    each parameter, such as a help message.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use."}
    )
    dataset_name: Optional[str] = field(
        default="customized", metadata={"help": "Should be \"customized\""}
    )
    is_custom_dataset: Optional[bool] = field(
        default=False, metadata={"help": "whether to use custom data"}
    )
    customized_cache_dir: Optional[str] = field(
        default=".cache/llm-ft/datasets",
        metadata={"help": "Where do you want to store the customized dataset caches"},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=1e10,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    disable_group_texts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether we group original samples together to generate sample"
                " sequences of length `block_size`. By default, we group every"
                " 1000 tokenized sequences together, divide them into "
                " [{total_num_tokens} / {block_size}] sequences, each with"
                " `block_size` tokens (the remaining tokens are ommited."
                " If this flag is set to True, we only group 1 tokenized"
                " sequence, i.e. cutting long sequence into chunks."
            )
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation File Path"},
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class FinetunerArguments(TrainingArguments):
    """
    Adapt transformers.TrainingArguments
    """
    pass


@dataclass
class EvaluatorArguments:
    """
    Define a class EvaluatorArguments using the dataclass decorator. The class contains several optional
    parameters that can be used to configure a evaluator.

    local_rank : str
        For distributed training: local_rank

    random_shuffle : bool

    use_wandb : bool

    random_seed : int, default = 1

    output_dir : str, default = './output_dir',

    mixed_precision : str, choice from ["bf16","fp16"].
        mixed precision mode, whether to use bf16 or fp16

    deepspeed : 
        Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already
        loaded json file as a dict
    """
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"
        }
    )

    random_shuffle: Optional[bool] = field(
        default=False, 
        metadata={"help": ""
        }
    )
    
    use_wandb: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When this flag is True, wandb will be enabled"
            )
        },
    )
    random_seed: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "used to set random seed"
            )
        },
    )
    output_dir: Optional[str] = field(
        default="./output_dir",
        metadata={"help": "Output path for the inferenced results"},
    )
    mixed_precision: Optional[str] = field(
        default="bf16",
        metadata={
            "help": (
                "mixed precision mode, whether to use bf16 or fp16"
            ),
            "choices": ["bf16","fp16"],
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    answer_type: Optional[str] = field(
        default="text",
        metadata={
            "help": (
                'Question type for answer extraction from the decoder output.'
                ' Supported types: \n'
                '   1) "multiple_choice", e.g. A, B, C, D, ...\n'
                '   2) "binary_choice", e.g. yes, no, maybe\n'
                '   3) "math", e.g. 1.0, -3.52\n'
                '   4) "text", e.g. "I think that it is okay"\n'
                '   5) Special treatment for several datasets\n'
                '     - "gsm8k"\n'
                '     - "svamp"\n'
                '     - "asdiv"\n'
                '     - "addsub"\n'
                '     - "singleeq"\n'
                '     - "multiarith"\n'
                '     - "aqua"\n'
                '     - "csqa"\n'
                '     - "strategyqa"\n'
                '     - "pubmedqa"\n'
                '     - "medmcqa"\n'
                '     - "usmle"\n'
            )
        },
    )
    prompt_structure: Optional[str] = field(
        default="{input}",
        metadata={
            "help": (
                'Prompt structure to facilitate prompt engineering during'
                ' inference. The model will receive'
                ' `prompt_structure.format(input=input)` as its input.'
            )
        },
    )


@dataclass
class InferencerArguments:
    """
    Define a class InferencerArguments using the dataclass decorator. The class contains several optional
    parameters that can be used to configure a inferencer.

    local_rank : str
        For distributed training: local_rank

    random_seed : int, default = 1

    deepspeed :
        Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already
        loaded json file as a dict
    mixed_precision : str, choice from ["bf16","fp16"].
        mixed precision mode, whether to use bf16 or fp16

    """
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"
        }
    )
    random_seed: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "used to set random seed"
            )
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    mixed_precision: Optional[str] = field(
        default="bf16",
        metadata={
            "help": (
                "mixed precision mode, whether to use bf16 or fp16"
            ),
            "choices": ["bf16","fp16"],
        },
    )


PIPELINE_ARGUMENT_MAPPING = {
    "finetuner": FinetunerArguments,
    "evaluator": EvaluatorArguments,
    "inferencer": InferencerArguments,
}


class AutoArguments:
    """
    Automatically choose arguments from FinetunerArguments or EvaluatorArguments.
    """
    def get_pipeline_args_class(pipeline_name: str):
        return PIPELINE_ARGUMENT_MAPPING[pipeline_name]
