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
    "float_only": ["value"],
}
