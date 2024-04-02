# Dataset

We provide several available datasets under `data`. You may download them all by running: 
```sh
cd data && ./download.sh all && cd -
```
You can replace `all` with a specific dataset name to only download that dataset (e.g. `./download.sh alpaca`).

Customized datasets are strongly encouraged, since this way users can apply
their own prompt engineering techniques over various source datasets. As long
as the generated dataset following the format below, they can be accepted as
the input of our pipelines :hugs:


## Dataset Format in General

To specify the input for model finetune, users can provide a list of `.json`
files under a specified dataset directory. For example,

```sh
|- path_to_dataset
  |- data_1.json
  |- data_2.json
  |- another_data.json
  |- ...
```

For inference, we currently only support a single `.json` file.

Each json file shall have the following format (three instances with four keys
for example),

```json
{
  "type": "TYPE",
  "instances": [
    {
        "KEY_1": "VALUE_1.1",
        "KEY_2": "VALUE_1.2",
        "KEY_3": "VALUE_1.3",
        "KEY_4": "VALUE_1.4",
    },
    {
        "KEY_1": "VALUE_2.1",
        "KEY_2": "VALUE_2.2",
        "KEY_3": "VALUE_2.3",
        "KEY_4": "VALUE_2.4",
    },
    {
        "KEY_1": "VALUE_3.1",
        "KEY_2": "VALUE_3.2",
        "KEY_3": "VALUE_3.3",
        "KEY_4": "VALUE_3.4",
    },
  ]
}
```

where the `TYPE` indicates the dataset type and defines the set of keys
`{ KEY_1, KEY_2, ... }` and their corresponding interpretations. The list of
supported types are listed as follows.

## Supported Dataset and Detailed Formats

### TextOnly

This is the most common dataset type, which only contains raw texts in each
sample. This type of dataset can be used as the training set for text decoder
models, or the input of decoder models / encoder-decoder models. Its format is
as follows (three instances for example),

```json
{
  "type": "text_only",
  "instances": [
    {  "text": "SAMPLE_TEXT_1" },
    {  "text": "SAMPLE_TEXT_2" },
    {  "text": "SAMPLE_TEXT_3" },
  ]
}
```

For example, `data/example_dataset/train/train_50.json` has the aboved format.

### Text2Text

This is the dataset type mostly used for inferencing, which contains a pair of
texts in each sample. This type of dataset can be used as the training set for
text encoder-decoder models, or question-answer pair for evaluating model
inferences. Its format is as follows (three instances for example),

```json
{
  "type": "text2text",
  "instances": [
    {
        "input": "SAMPLE_INPUT_1",
        "output": "SAMPLE_OUTPUT_1",
    },
    {
        "input": "SAMPLE_INPUT_2",
        "output": "SAMPLE_OUTPUT_2",
    },
    {
        "input": "SAMPLE_INPUT_3",
        "output": "SAMPLE_OUTPUT_3",
    },
  ]
}
```

For example, `data/example_dataset/test/test_13.json` has the aboved format.

### Conversation

```{admonition} **Work in Progress**
:class: info

We are rapidly working on this data format.  
```

Conversational data are commonly used in sft process. We currently support conversational data in ShareGPT format:
```json
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
```
Tips:
- [Please see the section below] <del>Users don't have to apply any model-specific input formatting such as system prompt, tool description, and instruction template, as the pipeline will handle the conversion automatically based on different models. If the dataset is already formatted in user-assistant pairs with system prompt and model-specific instruction format, one can convert that dataset into correct lmflow conversation json format, leaving `system` and `tools` empty. </del> 
- `conversation_id` is only for convience of tracking the conversation and will not be used in the pipeline. Users can set it to any value.
- `system` and `tools` are not required. Setting them to empty string `""` and empty list `[""]` respectively when not applicable.
- Please make sure the messages are:
  1. Start with an user message.
  2. In the correct order. The pipeline will not check the order of the messages.
  3. In pairs of user and assistant (i.e., the length of the messages should be even). If the conversation ends with the user, the pipeline will trim the last user message.

```{admonition} Auto Formatting
:class: warning

Auto formatting is not up currently. In other word, users need to include their system prompt and tool prompt into the first message, and apply the instruction template to user inputs manually. For example, for Llama-2-Chat:
```json
{
  "type": "conversation",
  "instances": [
    {
      "conversation_id": 1,
      "system": "",
      "tools": [""],
      "messages": [
        {
            "role": "user",
            "content": "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello! [/INST]"
        },
        {
            "role": "assistant",
            "content": "Hi, how are you?"
        },
        {
            "role": "user",
            "content": "[INST] Good. [/INST]"
        },
        {
            "role": "assistant",
            "content": "Glad to hear that."
        }
      ]
    },
    {
      "conversation_id": 2,
      "system": "",
      "tools": [""],
      "messages": [
        {
            "role": "user",
            "content": "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nWhat's the weather like now? [/INST]"
        },
        {
            "role": "assistant",
            "content": "I'm sorry for any confusion, but as an AI, I don't have access to real-time data such as current weather conditions."
        }
      ]
    }
  ]
}
```