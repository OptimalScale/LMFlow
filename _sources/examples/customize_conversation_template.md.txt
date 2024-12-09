# Customize Conversation Template

> For beginners: Why template?   
> Almost all LLMs today do a simple job - predict the next "word". To make the interaction between user and model smoother, developers use tricks: they add special "words" to the input text (at back-end, thus invisible to the user when using services like ChatGPT) to "tell" the model what user had said before, and ask the model to respond like an assistant. These "hidden words" are called "template".

We provide the flexibility to customize the conversation template. You can customize your own conversation template by following the steps below:

### 1. Decompose your conversations
Say you want to make the conversations between user and assistant look like:  

```
<bos>System:
You are a chatbot developed by LMFlow team.

User:
Who are you?

Assistant:
I am a chatbot developed by LMFlow team.<eos>

User:
How old are you?

Assistant:
I don't age like humans do. I exist as a piece of software, so I don't have a concept of age in the traditional sense.<eos>
```

It is easy to abstract the format for each message:
- System message: `System:\n{{content}}\n\n`  
- User message: `User:\n{{content}}\n\n`  
- Assistant message: `Assistant:\n{{content}}\n\n<eos>`  

Also, we have a bos token at the beginning of the conversation session.

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

    # Similar to the special starter... (just for illustration, commented out 
    # since it is not necessary for our purposed template above)
    # special_stopper=TemplateComponent(type='token', content='eos_token')
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