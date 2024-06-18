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
    "float_only": ["value"],
    "image_text": ["images", "text"],
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
# NOTE: Be careful, when passing lora_target_modules through arg parser, the 
# value should be like'--lora_target_modules q_proj, v_proj \', while specifying
# here, it should be in list format.
LMFLOW_LORA_TARGET_MODULES_MAPPING = {
    'qwen2': ["q_proj", "v_proj"],
    'internlm2': ["wqkv"],
}

# vllm inference
MEMORY_SAFE_VLLM_INFERENCE_FINISH_FLAG = "MEMORY_SAFE_VLLM_INFERENCE_DONE"