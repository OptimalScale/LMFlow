# Customize Conversation Template

```{admonition} **Work in Progress**
:class: info

We are rapidly working on this page.  
```

> For beginners: Why template?   
> Almost all LLMs today do a simple job - predict the next "word". To make the interaction between user and model smoother, developers use tricks: they add special "words" to the input text (at back-end, thus invisible to the user when using services like ChatGPT) to "tell" the model what user had said before, and ask the model to respond like an assistant. These "hidden words" are called "template".

We provide the flexibility to customize the conversation template. You can customize your own conversation template by following the steps below:

## Knowing the conversation template of your model

The conversation template varies according to the model you are using. For example:  

The template for Llama-2 looks like:  
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST] {{assistant_reply_0}}</s>
```

Find more templates [here](./supported_conversation_template.md).


## Make your own template

`TemplateComponent`s to a conversation template is just like bricks to a LEGO house. You can build your own template by combining different components.

The following provides an example of building a conversation template for the ChatML format:

### 1. Decompose the official template
The official template looks like:
```
<|im_start|>system\n{{system_message}}<|im_end|>\n<|im_start|>user\n{{user_message_0}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_0}}<|im_end|>\n<|im_start|>user\n{{user_message_1}}<|im_end|>\n<|im_start|>assistant\n{{assistant_reply_1}}<|im_end|>\n
```
It is easy to recognize the format for each message:
- System message: `<|im_start|>system\n{{system_message}}<|im_end|>\n`  
- User message: `<|im_start|>user\n{{user_message}}<|im_end|>\n`  
- Assistant message: `<|im_start|>assistant\n{{assistant_reply}}<|im_end|>\n`  

###  2. Choose proper `Formatter`  
Recall the requirements for a conversation dataset:  
> - `system`: `Optional[string]`. 
> - `tools`: `Optional[List[string]]`.  
> - `messages`: `List[Dict]`.  
>    - `role`: `string`.  
>    - `content`: `string`.  

System message, user message, and assistant message are strings thus we can use `StringFormatter` for them.

### 3. Build the template
```python
from dataclasses import dataclass

from lmflow.utils.conversation_formatter import Formatter, TemplateComponent, StringFormatter
from lmflow.utils.conversation_template import ConversationTemplate


@dataclass
class ChatMLConversationTemplate(ConversationTemplate):
    user_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>user\n{{content}}<|im_end|>\n')
        ]
    )
    assistant_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    )
    system_formatter: Formatter = StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>system\n{{content}}<|im_end|>\n')
        ]
    )
```