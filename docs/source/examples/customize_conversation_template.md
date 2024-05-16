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
All preset templates are located at `src/lmflow/utils/conversation_template`.

Within the template file, define your own template like:

```python
from .base import StringFormatter, TemplateComponent, ConversationTemplate


YOUR_TEMPLATE = ConversationTemplate(
    template_name='your_template_name',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='User:\n{{content}}\n\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='Assistant:\n{{content}}\n\n'),
            TemplateComponent(type='token', content='eos_token') # this will add the eos token at the end of every assistant message
            # please refer to the docstring of the `TemplateComponent` class to 
            # see the difference between different types of components.
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='System:\n{{content}}\n\n')
        ]
    )
    # For models that has ONLY ONE bos token at the beginning of 
    # a conversation session (not a conversation pair), user can
    # specify a special starter to add that starter to the very
    # beginning of the conversation session. 
    # eg:
    #   llama-2: <s> and </s> at every pair of conversation 
    #   v.s.
    #   llama-3: <|begin_of_text|> only at the beginning of a session
    special_starter=TemplateComponent(type='token', content='bos_token'),
    # Similar to the special starter...
    special_stopper=TemplateComponent(type='token', content='eos_token')

)
```

Feel free to create your own template by inheriting the `ConversationTemplate` class. Llama-2 v.s. llama-3 would be a good examples to refer to.

### 4. Register your template
After defining your own template, you need to register it in the `src/lmflow/utils/conversation_template/__init__.py` file. 

```python
# ...
from .your_template_file import YOUR_TEMPLATE


PRESET_TEMPLATES = {
    #...
    'your_template_name': YOUR_TEMPLATE,
}
```

### 5. Use your template
You are all set! Specify the template name in, for example, your finetune script:

```bash
./scripts/run_finetune.sh \
    --model_name_or_path path_to_your_model \
    --dataset_path your_conversation_dataset \
    --conversation_template your_template_name \
    --output_model_path output_models/your_model
```