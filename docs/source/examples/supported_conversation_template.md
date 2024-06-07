# Supported Conversation Template

- [Supported Conversation Template](#supported-conversation-template)
  - [ChatGLM-3](#chatglm-3)
  - [ChatML](#chatml)
  - [DeepSeek](#deepseek)
  - [Gemma](#gemma)
  - [InternLM2](#internlm2)
  - [Llama-2](#llama-2)
  - [Llama-3](#llama-3)
  - [Mixtral 8x22B](#mixtral-8x22b)
  - [Mixtral 8x7B](#mixtral-8x7b)
  - [Phi-3](#phi-3)
  - [Qwen-2](#qwen-2)
  - [Yi](#yi)
  - [Yi-1.5](#yi-15)
  - [Zephyr](#zephyr)


## ChatGLM-3
**With a system message**
```
[gMASK]sop<|system|>\n {{system_message}}<|user|>\n {{user_message_0}}
```

**Without a system message**
```
[gMASK]sop<|user|>\n {{user_message_0}}
```

**A complete conversation**
```
[gMASK]sop<|system|>\n {{system_message}}<|user|>\n {{user_message_0}}<|assistant|>\n {{assistant_reply_0}}
```

**Multiple rounds**
```
[gMASK]sop<|system|>\n {{system_message}}<|user|>\n {{user_message_0}}<|assistant|>\n {{assistant_reply_0}}<|user|>\n {{user_message_1}}<|assistant|>\n {{assistant_reply_1}}
```

**jinja template**  
[[Reference](https://huggingface.co/THUDM/chatglm3-6b/blob/103caa40027ebfd8450289ca2f278eac4ff26405/tokenizer_config.json#L42)]
```
{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}
```

**Filled Example**
```
[gMASK]sop<|system|>\n You are a chatbot developed by LMFlow team.<|user|>\n Who are you?<|assistant|>\n I am a chatbot developed by LMFlow team.<|user|>\n How old are you?<|assistant|>\n I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.
```


## ChatML
**With a system message**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**Without a system message**
```
<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**A complete conversation**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n
```

**Multiple rounds**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```

**jinja template**  
[[Reference](https://huggingface.co/mlabonne/OrpoLlama-3-8B/blob/3534d0562dee3a541d015ef908a71b0aa9085488/tokenizer_config.json#L2073)]
```
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**Filled Example**
```
<|im_start|>system\nYou are a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nHow old are you?<|im_end|>\n<|im_start|>assistant\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|im_end|>\n
```


## DeepSeek
**With a system message** 
```
<｜begin▁of▁sentence｜>{{system_message}}\n\nUser: {{user_message_0}}\n\n
```

**Without a system message**
```
<｜begin▁of▁sentence｜>User: {{user_message_0}}\n\n
```

**A complete conversation**
```
<｜begin▁of▁sentence｜>{{system_message}}\n\nUser: {{user_message_0}}\n\nAssistant: {{assistant_reply_0}}<｜end▁of▁sentence｜>
```

**Multiple rounds**
```
<｜begin▁of▁sentence｜>{{system_message}}\n\nUser: {{user_message_0}}\n\nAssistant: {{assistant_reply_0}}<｜end▁of▁sentence｜>User: {{user_message_1}}\n\nAssistant: {{assistant_reply_1}}<｜end▁of▁sentence｜>
```

**jinja template**  
[[Reference](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat/blob/941577e8236164bc96829096d20c61568630d7bc/tokenizer_config.json#L34)]
```
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}
```

**Filled Example**
```
<｜begin▁of▁sentence｜>You are a chatbot developed by LMFlow team.\n\nUser: Who are you?\n\nAssistant: I am a chatbot developed by LMFlow team.<｜end▁of▁sentence｜>User: How old are you?\n\nAssistant: I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<｜end▁of▁sentence｜>
```


## Gemma
**With a system message** 
```{admonition} NOTICE
:class: warning

As of now, Gemma does not support system messages officially. `ConversationTemplate` will add your system messages right after the bos token and before the user message without any special formatting. For more details, please refer to the [official template](https://huggingface.co/google/gemma-1.1-2b-it/blob/bf4924f313df5166dee1467161e886e55f2eb4d4/tokenizer_config.json#L1507).
```
```
<bos>{{system_message}}<start_of_turn>user\n{{user_message_0}}<end_of_turn>\n
```

**Without a system message**
```
<bos><start_of_turn>user\n{{user_message_0}}<end_of_turn>\n
```

**A complete conversation**
```
<bos>{{system_message}}<start_of_turn>user\n{{user_message_0}}<end_of_turn>\n<start_of_turn>model\n{{assistant_reply_0}}<end_of_turn>\n
```

**Multiple rounds**
```
<bos>{{system_message}}<start_of_turn>user\n{{user_message_0}}<end_of_turn>\n<start_of_turn>model\n{{assistant_reply_0}}<end_of_turn>\n<start_of_turn>user\n{{user_message_1}}<end_of_turn>\n<start_of_turn>model\n{{assistant_reply_1}}<end_of_turn>\n
```

**jinja template**  
[[Reference](https://huggingface.co/google/gemma-1.1-2b-it/blob/bf4924f313df5166dee1467161e886e55f2eb4d4/tokenizer_config.json#L1507)]
```
{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}
```

**Filled Example**
```
<bos>You are a chatbot developed by LMFlow team.<start_of_turn>user\nWho are you?<end_of_turn>\n<start_of_turn>model\nI am a chatbot developed by LMFlow team.<end_of_turn>\n<start_of_turn>user\nHow old are you?<end_of_turn>\n<start_of_turn>model\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<end_of_turn>\n
```


## InternLM2  
**With a system message** 
```
<s><|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**Without a system message**
```
<s><|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**A complete conversation**
```
<s><|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n
```

**Multiple rounds**
```
<s><|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```

**jinja template**  
[[Reference](https://huggingface.co/internlm/internlm2-chat-20b/blob/477d4748322a8a3b28f62b33f0f6dd353cd0b66d/tokenizer_config.json#L93)]
```
{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**Filled Example**
```
<s><|im_start|>system\nYou are a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nHow old are you?<|im_end|>\n<|im_start|>assistant\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|im_end|>\n
```


## Llama-2
**With a system message** 
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST]
```

**Without a system message**
```
<s>[INST] {{user_message_0}} [/INST]
```

**A complete conversation**
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST] {{assistant_reply_0}}</s>
```

**Multiple rounds**
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST] {{assistant_reply_0}}</s><s>[INST] {{user_message_1}} [/INST] {{assistant_reply_1}}</s>
```

**jinja template**  
[[Reference](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/tokenizer_config.json#L12)]
```
{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
```

**Filled Example**
```
<s>[INST] <<SYS>>\nYou are a chatbot developed by LMFlow team.\n<</SYS>>\n\nWho are you? [/INST] I am a chatbot developed by LMFlow team.</s><s>[INST] How old are you? [/INST] I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.</s>
```


## Llama-3
**With a system message**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{system_message}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{user_message_0}}<|eot_id|>
```

**Without a system message**
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{{user_message_0}}<|eot_id|>
```

**A complete conversation**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{system_message}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{user_message_0}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{assistant_reply_0}}<|eot_id|>
```

**Multiple rounds**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{system_message}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{user_message_0}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{assistant_reply_0}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{user_message_1}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{assistant_reply_1}}<|eot_id|>
```

**jinja template**  
[[Reference](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/2b724926966c141d5a60b14e75a5ef5c0ab7a6f0/tokenizer_config.json#L2053)]
```
{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
```

**Filled Example**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a chatbot developed by LMFlow team.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI am a chatbot developed by LMFlow team.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow old are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|eot_id|>
```


## Mixtral 8x22B
```{admonition} **Work in Progress**
:class: info

This template is not preseted in LMFlow currently. We are working on it and will update it soon.  
```

```{admonition} NOTICE
:class: warning

The conversation template for Mixtral 8x22B is slightly different from the template for Mixtral 8x7B.
```
**jinja template**  
[[Reference](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)]
```
{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
```


## Mixtral 8x7B
```{admonition} **Work in Progress**
:class: info

This template is not preseted in LMFlow currently. We are working on it and will update it soon.  
```

```{admonition} NOTICE
:class: warning

The conversation template for Mixtral 8x7B is slightly different from the template for Mixtral 8x22B.
```
**jinja template**  
[[Reference](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/1e637f2d7cb0a9d6fb1922f305cb784995190a83/tokenizer_config.json#L42)]
```
{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
```


## Phi-3
**With a system message**
```
<s><|system|>\n{{system_message}}<|end|>\n<|user|>\n{{user_message_0}}<|end|>\n<|endoftext|>
```

**Without a system message**
```
<s><|user|>\n{{user_message_0}}<|end|>\n<|endoftext|>
```

**A complete conversation**
```
<s><|system|>\n{{system_message}}<|end|>\n<|user|>\n{{user_message_0}}<|end|>\n<|assistant|>\n{{assistant_reply_0}}<|end|>\n<|endoftext|>
```

**Multiple rounds**
```
<s><|system|>\n{{system_message}}<|end|>\n<|user|>\n{{user_message_0}}<|end|>\n<|assistant|>\n{{assistant_reply_0}}<|end|>\n<|user|>\n{{user_message_1}}<|end|>\n<|assistant|>\n{{assistant_reply_1}}<|end|>\n<|endoftext|>
```

**jinja template**  
[[Reference]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/3a811845d89f3c1b3f41b341d0f9f05104769f35/tokenizer_config.json#L338)
```
{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}
```

**Filled Example**
```
<s><|system|>\nYou are a chatbot developed by LMFlow team.<|end|>\n<|user|>\nWho are you?<|end|>\n<|assistant|>\nI am a chatbot developed by LMFlow team.<|end|>\n<|user|>\nHow old are you?<|end|>\n<|assistant|>\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|end|>\n<|endoftext|>
```


## Qwen-2

(Also Qwen-1.5)

**With a system message**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**Without a system message**
```
<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**A complete conversation**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n
```

**Multiple rounds**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```

**jinja template**  
[[Reference](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/1af63c698f59c4235668ec9c1395468cb7cd7e79/tokenizer_config.json#L31)]
```
{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**Filled Example**
```
<|im_start|>system\nYou are a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nHow old are you?<|im_end|>\n<|im_start|>assistant\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|im_end|>\n
```


## Yi
**With a system message**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**Without a system message**
```
<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**A complete conversation**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n
```

**Multiple rounds**
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```

**jinja template**  
[[Reference](https://huggingface.co/01-ai/Yi-34B-Chat/blob/c556c018b58980fb651ff4952d86cd5250a713d0/tokenizer_config.json#L60)]
```
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**Filled Example**
```
<|im_start|>system\nYou are a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nHow old are you?<|im_end|>\n<|im_start|>assistant\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|im_end|>\n
```


## Yi-1.5
**With a system message**
```
{{system_message}}<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**Without a system message**
```
<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

**A complete conversation**
```
{{system_message}}<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n
```

**Multiple rounds**
```
{{system_message}}<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```

**jinja template**  
[[Reference](https://huggingface.co/01-ai/Yi-1.5-6B-Chat/blob/d68dab90947a3c869e28c9cb2806996af99a6080/tokenizer_config.json#L40)]
```
{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}
```

**Filled Example**
```
You are a chatbot developed by LMFlow team.<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nHow old are you?<|im_end|>\n<|im_start|>assistant\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|im_end|>\n
```


## Zephyr
**With a system message**
```
<|system|>\n{{system_message}}</s>\n<|user|>\n{{user_message_0}}</s>\n
```

**Without a system message**
```
<|user|>\n{{user_message_0}}</s>\n
```

**A complete conversation**
```
<|system|>\n{{system_message}}</s>\n<|user|>\n{{user_message_0}}</s>\n<|assistant|>\n{{assistant_reply_0}}</s>\n
```

**Multiple rounds**
```
<|system|>\n{{system_message}}</s>\n<|user|>\n{{user_message_0}}</s>\n<|assistant|>\n{{assistant_reply_0}}</s>\n<|user|>\n{{user_message_1}}</s>\n<|assistant|>\n{{assistant_reply_1}}</s>\n
```

**jinja template**  
[[Reference](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta/blob/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473/tokenizer_config.json#L34)]
```
{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}
```

**Filled Example**
```
<|system|>\nYou are a chatbot developed by LMFlow team.</s>\n<|user|>\nWho are you?</s>\n<|assistant|>\nI am a chatbot developed by LMFlow team.</s>\n<|user|>\nHow old are you?</s>\n<|assistant|>\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.</s>\n
```