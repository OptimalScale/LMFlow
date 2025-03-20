#!/usr/bin/env python
# coding=utf-8
"""
Commonly used constants.
"""

TEXT_ONLY_DATASET_DESCRIPTION = (
"""
"text_only": a dataset with only raw text instances, with following format:

    {
        "type": "text_only",
        "instances": [
            { "text": "TEXT_1" },
            { "text": "TEXT_2" },
            ...
        ]
    }
"""
).lstrip("\n")


TEXT_TO_SCORED_TEXTLIST_DATASET_DESCRIPTION = (
"""
This kind of dataset is commonly used in reward model training/prediction, as well as rl training.
{
    "type": "text_to_scored_textlist",
    "instances": [
        {
            "input": "what's your name?",
            "output": [
                {"score": 1.0, "text": "My name is John"},
                {"score": -0.8, "text": "I'm John"}
            ]
        },
        {
            "input": "Who are you?",
            "output": [
                {"score": 1.5, "text": "My name is Amy"},
                {"score": 1.0, "text": "I'm Amy"}
            ]
        },
    ]
}
"""
).lstrip("\n")


PAIRED_TEXT_TO_TEXT_DATASET_DESCRIPTION = (
"""
This kind of dataset is commonly used in reward model training as well as rl training.
{
    "type": "paired_text_to_text",
    "instances": [
        {
            "prompt": "Who are you?",
            "chosen": "My name is Amy.",
            "rejected": "I'm Amy",
            "margin": 0.6
        },
        {
            "prompt": "what's your name?",
            "chosen": "My name is John.",
            "rejected": "I'm John",
            "margin": 0.5
        }
    ]
}
"""
).lstrip("\n")


TEXT_ONLY_DATASET_DETAILS = (
"""
    For example,

    ```python
    from lmflow.datasets import Dataset

    data_dict = {
        "type": "text_only",
        "instances": [
            { "text": "Human: Hello. Bot: Hi!" },
            { "text": "Human: How are you today? Bot: Fine, thank you!" },
        ]
    }
    dataset = Dataset.create_from_dict(data_dict)
    ```

    You may also save the corresponding format to json,
    ```python
    import json
    from lmflow.args import DatasetArguments
    from lmflow.datasets import Dataset

    data_dict = {
        "type": "text_only",
        "instances": [
            { "text": "Human: Hello. Bot: Hi!" },
            { "text": "Human: How are you today? Bot: Fine, thank you!" },
        ]
    }
    with open("data.json", "w") as fout:
        json.dump(data_dict, fout)

    data_args = DatasetArgument(dataset_path="data.json")
    dataset = Dataset(data_args)
    new_data_dict = dataset.to_dict()
    # `new_data_dict` Should have the same content as `data_dict`
    ```
"""
).lstrip("\n")


TEXT2TEXT_DATASET_DESCRIPTION = (
"""
"text2text": a dataset with input & output instances, with following format:

    {
        "type": "text2text",
        "instances": [
            { "input": "INPUT_1", "output": "OUTPUT_1" },
            { "input": "INPUT_2", "output": "OUTPUT_2" },
            ...
        ]
    }
"""
).lstrip("\n")


CONVERSATION_DATASET_DESCRIPTION = (
"""
"conversation": a dataset with conversation instances, with following format (`conversation_id`, `system` and `tools` are optional):

    {
    "type": "conversation",
    "instances": [
        {
        "conversation_id": "CONVERSATION_ID",
        "system": "SYSTEM_PROPMT",
        "tools": ["TOOL_DESCRIPTION_1","TOOL_DESCRIPTION_2","TOOL_DESCRIPTION_X"],
        "messages": [
            {
                "role": "user",
                "content": "USER_INPUT_1"
            },
            {
                "role": "assistant",
                "content": "ASSISTANT_RESPONSE_1"
            },
            {
                "role": "user",
                "content": "USER_INPUT_2"
            },
            {
                "role": "assistant",
                "content": "ASSISTANT_RESPONSE_2"
            }
        ]
        },
        {
        "conversation_id": "CONVERSATION_ID",
        "system": "SYSTEM_PROPMT",
        "tools": ["TOOL_DESCRIPTION_1"],
        "messages": [
            {
                "role": "user",
                "content": "USER_INPUT_1"
            },
            {
                "role": "assistant",
                "content": "ASSISTANT_RESPONSE_1"
            }
        ]
        }
    ]
    }
"""
).lstrip("\n")


PAIRED_CONVERSATION_DATASET_DESCRIPTION = (
"""
"paired_conversation": a dataset with paired conversation instances, with following format:

    {
        "type": "paired_conversation",
        "instances": [
            {
                "chosen": {
                    "conversation_id": "CONVERSATION_ID",
                    "system": "SYSTEM_PROPMT",
                    "tools": ["TOOL_DESCRIPTION_1","TOOL_DESCRIPTION_2","TOOL_DESCRIPTION_3"],
                    "messages": [
                        {
                            "role": "user",
                            "content": "USER_INPUT_1"
                        },
                        {
                            "role": "assistant",
                            "content": "ASSISTANT_RESPONSE_1_GOOD"
                        },
                        {
                            "role": "user",
                            "content": "USER_INPUT_2"
                        },
                        {
                            "role": "assistant",
                            "content": "ASSISTANT_RESPONSE_2_GOOD"
                        }
                    ]
                },
                "rejected": {
                    "conversation_id": "CONVERSATION_ID",
                    "system": "SYSTEM_PROPMT",
                    "tools": ["TOOL_DESCRIPTION_1","TOOL_DESCRIPTION_2","TOOL_DESCRIPTION_3"],
                    "messages": [
                        {
                            "role": "user",
                            "content": "USER_INPUT_1"
                        },
                        {
                            "role": "assistant",
                            "content": "ASSISTANT_RESPONSE_1_BAD"
                        },
                        {
                            "role": "user",
                            "content": "USER_INPUT_2"
                        },
                        {
                            "role": "assistant",
                            "content": "ASSISTANT_RESPONSE_2_BAD"
                        }
                    ]
                }
            }
        ]
    }
"""
).lstrip("\n")


TEXT_TO_TEXTLIST_DATASET_DESCRIPTION = (
"""
This kind of dataset is commonly used in reward model inference.
{
    "type": "text_to_textlist",
    "instances": [
        {
            "input": "what's your name?",
            "output": [
                "My name is John",
                "I'm John",
            ]
        },
        {
            "input": "Who are you?",
            "output": [
                "My name is Amy",
                "I'm Amy",
            ]
        },
    ]
}
"""
).lstrip("\n")


TEXT2TEXT_DATASET_DETAILS = (
"""
    For example,

    ```python
    from lmflow.datasets import Dataset

    data_dict = {
        "type": "text2text",
        "instances": [
            {
                "input": "Human: Hello.",
                "output": "Bot: Hi!",
            },
            {
                "input": "Human: How are you today?",
                "output": "Bot: Fine, thank you! And you?",
            }
        ]
    }
    dataset = Dataset.create_from_dict(data_dict)
    ```

    You may also save the corresponding format to json,
    ```python
    import json
    from lmflow.args import DatasetArguments
    from lmflow.datasets import Dataset

    data_dict = {
        "type": "text2text",
        "instances": [
            {
                "input": "Human: Hello.",
                "output": "Bot: Hi!",
            },
            {
                "input": "Human: How are you today?",
                "output": "Bot: Fine, thank you! And you?",
            }
        ]
    }
    with open("data.json", "w") as fout:
        json.dump(data_dict, fout)

    data_args = DatasetArgument(dataset_path="data.json")
    dataset = Dataset(data_args)
    new_data_dict = dataset.to_dict()
    # `new_data_dict` Should have the same content as `data_dict`
    ```
"""
).lstrip("\n")


FLOAT_ONLY_DATASET_DESCRIPTION = (
"""
"float_only": a dataset with only float instances, with following format:

    {
        "type": "float_only",
        "instances": [
            { "value": "FLOAT_1" },
            { "value": "FLOAT_2" },
            ...
        ]
    }
"""
).lstrip("\n")


TEXT_ONLY_DATASET_LONG_DESCRITION = (
    TEXT_ONLY_DATASET_DESCRIPTION + TEXT_ONLY_DATASET_DETAILS
)

TEXT2TEXT_DATASET_LONG_DESCRITION = (
    TEXT2TEXT_DATASET_DESCRIPTION + TEXT2TEXT_DATASET_DETAILS
)


DATASET_DESCRIPTION_MAP = {
    "text_only": TEXT_ONLY_DATASET_DESCRIPTION,
    "text2text": TEXT2TEXT_DATASET_DESCRIPTION,
    "float_only": FLOAT_ONLY_DATASET_DESCRIPTION,
}

INSTANCE_FIELDS_MAP = {
    "text_only": ["text"],
    "text2text": ["input", "output"],
    "conversation": ["messages"], # system, tools and conversation_id are optional
    "paired_conversation": ["chosen", "rejected"],
    "paired_text_to_text": ["prompt", "chosen", "rejected"],
    "float_only": ["value"],
    "image_text": ["images", "text"],
    "text_to_textlist": ["input", "output"],
    "text_to_scored_textlist": ["input", "output"],
}

CONVERSATION_ROLE_NAMES = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "function": "function",
    "observation": "observation"
}

# LLAVA constants
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Lora
# NOTE: This work as a mapping for those models that `peft` library doesn't support yet, and will be 
# overwritten by peft.utils.constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# if the model is supported (see hf_model_mixin.py).
# NOTE: When passing lora_target_modules through arg parser, the 
# value should be a string. Using commas to separate the module names, e.g.
# "--lora_target_modules 'q_proj, v_proj'". 
# However, when specifying here, they should be lists.
LMFLOW_LORA_TARGET_MODULES_MAPPING = {
    'qwen2': ["q_proj", "v_proj"],
    'internlm2': ["wqkv"],
    'hymba': ["x_proj.0", "in_proj", "out_proj", "dt_proj.0"]
}

# vllm inference
MEMORY_SAFE_VLLM_INFERENCE_FINISH_FLAG = "MEMORY_SAFE_VLLM_INFERENCE_DONE"
RETURN_CODE_ERROR_BUFFER = [
    134
]
# return code 134:
# > Fatal Python error: _enter_buffered_busy: could not acquire lock for <_io.BufferedWriter name='<stdout>'> 
# > at interpreter shutdown, possibly due to daemon threads
# The above error, by our observation, is due to the kill signal with unfinished 
# stdout/stderr writing in the subprocess
MEMORY_SAFE_VLLM_INFERENCE_ENV_VAR_TO_REMOVE = [
    "OMP_NUM_THREADS",
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
]

# dpov2 align
MEMORY_SAFE_DPOV2_ALIGN_ENV_VAR_TO_REMOVE = [
    "OMP_NUM_THREADS",
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
]