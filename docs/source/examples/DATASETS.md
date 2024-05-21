# Dataset

- [Dataset](#dataset)
  - [Dataset Format in General](#dataset-format-in-general)
  - [Supported Dataset and Detailed Formats](#supported-dataset-and-detailed-formats)
    - [Conversation](#conversation)
      - [Data Format](#data-format)
      - [Conversation Template](#conversation-template)
      - [Customize Conversation Template](#customize-conversation-template)
    - [TextOnly](#textonly)
    - [Text2Text](#text2text)
    - [Paired Conversation](#paired-conversation)

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

### Conversation

#### Data Format

Conversational data are commonly used in sft process. We currently support conversational data in ShareGPT format:

````{dropdown} A conversation dataset
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
````
Data types:
- `conversation_id`: `Optional[Any]`. An identifier for the conversation. `conversation_id` is only for convience of tracking the conversation and will not be used in the pipeline.
- `system`: `Optional[string]`. A system prompt that is used to start the conversation.
- `tools`: `Optional[List[string]]`. A list of tools that are used in the conversation.
- `messages`: `List[Dict]`. A list of messages in the conversation. Each message contains the following fields:
  - `role`: `string`. The role of the message. It can be either `user` or `assistant`.
  - `content`: `string`. The content of the message.

> We are working on supporting customized message keys and role names. Please stay tuned.

Tips:
- Please make sure the messages are:
  1. Start with an user message.
  2. In the correct order. The pipeline will not check the order of the messages.
  3. In pairs of user and assistant (i.e., the length of the messages should be even). If the conversation ends with the user, the pipeline will trim the last user message.
  4. Make sure the `content`s are not empty. If the `content` is empty, the pipeline will add a space to it.

#### Conversation Template

Conversations should be formatted before feeding into the model. As of now, we've preset the conversation template for following models:

| Template Name | Filled Example | Detailed Template |
| ------------- | -------------- | ----------------- |
| `chatglm3` | `[gMASK]sop<\|system\|>`<br>` You are a chatbot developed by LMFlow team.<\|user\|>`<br>` Who are you?<\|assistant\|>`<br>` I am a chatbot developed by LMFlow team.<\|user\|>`<br>` How old are you?<\|assistant\|>`<br>` I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.` | [Link](./supported_conversation_template.md#chatglm3) |
| `chatml` | `<\|im_start\|>system`<br>`You are a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`Who are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I am a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`How old are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<\|im_end\|>`<br> | [Link](./supported_conversation_template.md#chatml) |
| `deepseek` | `<｜begin▁of▁sentence｜>You are a chatbot developed by LMFlow team.`<br><br>`User: Who are you?`<br><br>`Assistant: I am a chatbot developed by LMFlow team.<｜end▁of▁sentence｜>User: How old are you?`<br><br>`Assistant: I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<｜end▁of▁sentence｜>` | [Link](./supported_conversation_template.md#deepseek) |
| `gemma` | `<bos>You are a chatbot developed by LMFlow team.<start_of_turn>user`<br>`Who are you?<end_of_turn>`<br>`<start_of_turn>model`<br>`I am a chatbot developed by LMFlow team.<end_of_turn>`<br>`<start_of_turn>user`<br>`How old are you?<end_of_turn>`<br>`<start_of_turn>model`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<end_of_turn>`<br> | [Link](./supported_conversation_template.md#gemma) |
| `internlm2` | `<s><\|im_start\|>system`<br>`You are a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`Who are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I am a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`How old are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<\|im_end\|>`<br> | [Link](./supported_conversation_template.md#internlm2) |
| `llama3` | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>`<br><br>`You are a chatbot developed by LMFlow team.<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>`<br><br>`Who are you?<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>`<br><br>`I am a chatbot developed by LMFlow team.<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>`<br><br>`How old are you?<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>`<br><br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<\|eot_id\|>` | [Link](./supported_conversation_template.md#llama-3) |
| `llama2` | `<s>[INST] <<SYS>>`<br>`You are a chatbot developed by LMFlow team.`<br>`<</SYS>>`<br><br>`Who are you? [/INST] I am a chatbot developed by LMFlow team.</s><s>[INST] How old are you? [/INST] I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.</s>` | [Link](./supported_conversation_template.md#llama-2) |
| `phi3` | `<s><\|system\|>`<br>`You are a chatbot developed by LMFlow team.<\|end\|>`<br>`<\|user\|>`<br>`Who are you?<\|end\|>`<br>`<\|assistant\|>`<br>`I am a chatbot developed by LMFlow team.<\|end\|>`<br>`<\|user\|>`<br>`How old are you?<\|end\|>`<br>`<\|assistant\|>`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<\|end\|>`<br>`<\|endoftext\|>` | [Link](./supported_conversation_template.md#phi-3) |
| `qwen2` | `<\|im_start\|>system`<br>`You are a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`Who are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I am a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`How old are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<\|im_end\|>`<br> | [Link](./supported_conversation_template.md#qwen-2) |
| `yi` | Same as `chatml` | [Link](./supported_conversation_template.md#yi) |
| `yi1_5`| `You are a chatbot developed by LMFlow team.<\|im_start\|>user`<br>`Who are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I am a chatbot developed by LMFlow team.<\|im_end\|>`<br>`<\|im_start\|>user`<br>`How old are you?<\|im_end\|>`<br>`<\|im_start\|>assistant`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<\|im_end\|>`<br> | [Link](./supported_conversation_template.md#yi-15) |
| `zephyr` | `<\|system\|>`<br>`You are a chatbot developed by LMFlow team.</s>`<br>`<\|user\|>`<br>`Who are you?</s>`<br>`<\|assistant\|>`<br>`I am a chatbot developed by LMFlow team.</s>`<br>`<\|user\|>`<br>`How old are you?</s>`<br>`<\|assistant\|>`<br>`I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.</s>`<br> | [Link](./supported_conversation_template.md#zephyr) |

Passing the template name to the `--conversation_template` argument to apply the corresponding conversation template:
```sh
# scripts/run_finetune.sh
# ...
deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dataset_path ${dataset_path} \
    --conversation_template llama2 \
# ...
```

`````{admonition} Formatted Dataset
:class: info

For dataset that system prompts, tool prompts and templates are already applied (like the one below), user can run the finetune shell by passing `empty` or `empty_no_special_tokens` to the `--conversation_template` argument. `empty` template will add a bos token to the beginning of every round of conversation as well as a eos token to the end of every round of conversation. `empty_no_special_tokens` will not add any special tokens to the conversation, just concatenates the user and assistant messages. 
````{dropdown} A formatted dataset
```json
{
  "type": "conversation",
  "instances": [
    {
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
````
`````

#### Customize Conversation Template

Please refer to the [Customize Conversation Template](./customize_conversation_template.md) for more details.


### TextOnly

This is the most common dataset type, which only contains raw texts in each
sample. This type of dataset can be used as the training set for text decoder
models, or the input of decoder models / encoder-decoder models. Its format is
as follows (three instances for example),

````{dropdown} A textonly dataset
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
````

For example, `data/example_dataset/train/train_50.json` has the aboved format.


### Text2Text

This is the dataset type mostly used for inferencing, which contains a pair of
texts in each sample. This type of dataset can be used as the training set for
text encoder-decoder models, or question-answer pair for evaluating model
inferences. Its format is as follows (three instances for example):

````{dropdown} A text2text dataset
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
````

For example, `data/example_dataset/test/test_13.json` has the aboved format.


### Paired Conversation

```{admonition} **Work in Progress**
:class: info

We are working on paired conversation dataset and will update it soon.  
```

This type of dataset are commonly used for alignment such as [reward modeling](https://arxiv.org/abs/2203.02155), 
[Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), etc. For requirements of the conversations,
please refer to [conversation data](#conversation).

````{dropdown} A paired conversation dataset
```json
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
        },
        {
            "chosen": {
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
                        "content": "ASSISTANT_RESPONSE_1_GOOD"
                    }
                ]
            },
            "rejected": {
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
                        "content": "ASSISTANT_RESPONSE_1_BAD"
                    }
                ]
            }
        }
    ]
}
```
````