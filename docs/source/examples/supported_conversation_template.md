# Supported Conversation Template

## Table of Contents
- [Llama-2](#llama-2)
- [Qwen-2 (Qwen-1.5)](#qwen-2-qwen-15)

## Llama-2
### With a system message 
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST]
```

### Without a system message
```
<s>[INST] {{user_message_0}} [/INST]
```

### A complete conversation
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST] {{assistant_reply_0}}</s>
```

### Multiple rounds
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST] {{assistant_reply_0}}</s><s>[INST] {{user_message_1}} [/INST] {{assistant_reply_1}}</s>
```

### jinja template
```
work in progress
```

### Filled Example
```
<s>[INST] <<SYS>>\nYou are a chatbot developed by LMFlow team.\n<</SYS>>\n\nWho are you? [/INST] I am a chatbot developed by LMFlow team.</s><s>[INST] How old are you? [/INST] I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.</s>
```


## Qwen-2 (Qwen-1.5)
### With a system message
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

### Without a system message
```
<|im_start|>user\n{{user_message_0}}<|im_end|>\n
```

### A complete conversation
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n
```

### Multiple rounds
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```

### jinja template
```
work in progress
```

### Filled Example
```
<|im_start|>system\nYou are a Machine Learning expert.<|im_end|>\n<|im_start|>user\nIntroduce me the LISA fine-tuning technique.<|im_end|>\n<|im_start|>assistant\nLISA is short for Layerwise Importance Sampled AdamW. It is a novel and memory-efficient training strategy for large language models that outperforms existing methods like LoRA by selectively freezing layers during optimization.<|im_end|>\n<|im_start|>user\nWhere can I find the code?<|im_end|>\n<|im_start|>assistant\nLMFlow supports LISA. For more details, please refer to their official github repository.<|im_end|>\n
```