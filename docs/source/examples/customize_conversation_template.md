# Customize Conversation Template

```{admonition} **Work in Progress**
:class: info

We are rapidly working on this page.  
```

> For beginners: Why template?   
> Almost all LLMs today do a simple job - predict the next "word". To make the interaction between user and model smoother, developers use tricks: they add special "words" to the input text (at back-end, thus invisible to the user when using services like ChatGPT) to "tell" the model what user had said before, and ask the model to respond like an assistant. These "hidden words" are called "template".

We provide the flexibility to customize the conversation template. You can customize your own conversation template by following the steps below:

## Step 1: Knowing the conversation template of your model

The conversation template varies according to the model you are using. For example:  

The template for Llama-2 looks like:  
```
<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n{{user_message_0}} [/INST] {{assistant_reply_0}}</s>
```

Find more templates [here](./supported_conversation_template.md).


## Step 2: XXX
