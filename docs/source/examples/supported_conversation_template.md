# Supported Conversation Template

- [Supported Conversation Template](#supported-conversation-template)
  - [ChatGLM-3](#chatglm-3)
  - [ChatML](#chatml)
  - [DeepSeek](#deepseek)
  - [InternLM2](#internlm2)
  - [Llama-2](#llama-2)
  - [Llama-3](#llama-3)
  - [Mixtral 8x22B](#mixtral-8x22b)
  - [Mixtral 8x7B](#mixtral-8x7b)
  - [Phi-3](#phi-3)
  - [Qwen-2](#qwen-2)
  - [Yi](#yi)


## ChatGLM-3
```{admonition} **Work in Progress**
:class: info

This template is not preseted in LMFlow currently. We are working on it and will update it soon.  
```
**jinja template**  
[[Reference](https://huggingface.co/THUDM/chatglm3-6b/blob/103caa40027ebfd8450289ca2f278eac4ff26405/tokenizer_config.json#L42)]
```
{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}
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


## InternLM2
```{admonition} **Work in Progress**
:class: info

This template is not preseted in LMFlow currently. We are working on it and will update it soon.  
```
**jinja template**  
[[Reference](https://huggingface.co/internlm/internlm2-chat-20b/blob/477d4748322a8a3b28f62b33f0f6dd353cd0b66d/tokenizer_config.json#L93)]
```
{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
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
[[Reference](https://huggingface.co/Qwen/Qwen1.5-72B/blob/93bac0d1ae83d50c43b1793e2d74a00dc43a4c36/tokenizer_config.json#L31)]
```
{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**Filled Example**
```
<|im_start|>system\nYou are a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a chatbot developed by LMFlow team.<|im_end|>\n<|im_start|>user\nHow old are you?<|im_end|>\n<|im_start|>assistant\nI don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<|im_end|>\n
```


## Yi
```{admonition} **Work in Progress**
:class: info

This template is not preseted in LMFlow currently. We are working on it and will update it soon.  
```
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


