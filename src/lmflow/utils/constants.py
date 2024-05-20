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


PAIRED_CONVERSATION_DATASET_DESCRIPTION = (
"""
"paired_conversation": a dataset with paired conversation instances, with following format:

    {
    "type": "paired_conversation",
    "instances": [
        {
            "chosen": [
                {"content": "Please identify the top 5 rarest animals in the world.", "role": "user" }, 
                {"content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant" }, 
                {"content": "The ones that are really rare.", "role": "user" }, 
                {"content": "I’m not sure I can answer this question very easily. I’ll try to give you a basic overview of animals that are rare, and then you can tell me if you think that’s good enough. \n\nIf we were to measure animal populations by the area of land they occupy, the number 1 animal would probably be the Giant Panda. Panda populations are generally estimated to be about 2.1 million Pandas in the wild, which is a pretty small number of animals when we’re thinking about Earth’s entire surface area. The number 2 animal would probably be the Great Auk. Great Auks used to live in North America, and their total population was probably in the low hundreds, but they are now extinct.", "role": "assistant"}
            ],
            "rejected": [
                {"content": "Please identify the top 5 rarest animals in the world.", "role": "user"}, 
                {"content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant"}, 
                {"content": "The ones that are really rare.", "role": "user"}, 
                {"content": "Alright, here’s what I found:", "role": "assistant"}
            ]
        },
        {
            "chosen": [
                {"content": "How do I teach my kids to make forts at home?", "role": "user"}, 
                {"content": "Do you mean you want to teach them to build a simple structure they can hide in? Or something more elaborate?  I don’t know exactly what you mean by a “fort”, but maybe this gives you some ideas?", "role": "assistant"}
            ],
            "rejected": [
                {"content": "How do I teach my kids to make forts at home?", "role": "user"}, 
                {"content": "There are many ways to make forts!  The simplest might be to just pile up all the furniture in one room.  This works if you have a single room in your house, although it might still be a bit challenging to build a taller and sturdier fort this way.  The best way to build forts is to start with something easy, and then have fun exploring how you can improve it.", "role": "assistant"}
            ]
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
