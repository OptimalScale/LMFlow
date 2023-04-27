# Reward Modeling

## Introduction

Reinforcement Learning from Human Feedback (RLHF) requires a reward function to guide the adjustment of the generative model. In this example, we show how to use LMFlow framework to train a reward model following the procedure in the InstructGPT paper: https://arxiv.org/abs/2203.02155 . We use the Dahoas/full-hh-rlhf dataset as an example, where each sample of this dataset consists of a prompt and two responses from the assistant. In particular, the response with label "chosen" is preferred as compared to the response with label "rejected". The dataset consists of 112K training samples and 12.5K test samples. The following is an example sample of the dataset:

Prompt: 

" Human: What kind of noises did dinosaurs make? Assistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made would be Human: yes they did Assistant: to guess, and that would probably require lots of reading and a certain amount of imagination, so we’re not really prepared to do that. Human: you cant read Assistant: 

Chosen response: "You can read?"

Rejected response: "there’s a lot of stuff humans don’t know"

As an example, we prepare 10K sft training samples, 12K reward modeling samples (where 10% of them are split for evaluation) at ./data/hh_rlhf.

## Step 1 Supervised Finetuning (SFT)

We prepare the dataset used for supervised finetuning by adding a prefix to the Human and Assistant inputs to prompt model responses and simplify post-processing. Here is an example of a two-sample dataset to illustrate this.


```sh
{"type": "text_only", "instances": [{"text": "###Human: Should you buy a case to protect your cell phone?###Assistant: It depends on your circumstances.  If you carry your phone in a pocket or a purse then you probably want a case.  But if you only need a phone for quick interactions, a case may actually cause more harm than good.  What do you need the phone for?  Are you a parent, or do you work from home?###Human: What harm could it do?###Assistant: A phone case can damage the screen, for one thing.  It can also get you in trouble if you have your phone turned off for some reason.  Then you will turn it back on and it won’t do anything.  If you can afford to replace it, then you need a case to protect it.  The problem is that most people aren’t able to afford to replace their phones all the time.###Human: Thanks for letting me know.###Assistant: You’re welcome."}, {"text": "###Human: I'm trying to learn about the salam witch trials###Assistant: If you’re looking for books about witchcraft trials, I can recommend some books for you. But you probably mean you’re looking for more historical information about the Salem witch trials in 1692, and specifically about the salam witch trials in 1692?###Human: What are some good books to learn about the salam witch trials###Assistant: What would you like to learn?  If you’re interested in history, one of the best books is The Witch Persecutions in Colonial America: A History.  If you’re interested in witchcraft as a cultural phenomenon, you might enjoy two excellent books: Religion and the Decline of Magic: Studies in Popular Beliefs in Sixteenth- and Seventeenth-Century England by Keith Thomas and Magic, Witchcraft, and the Otherworld: An Anthropology of Superstition by Jack Goody.  If you’re interested in history specifically as it relates to religion, you might enjoy The Popish Plot, or Prelates' Plot: A History of the Popish Plot in England, by K. J. Everett."}]}
```

You can edit /scripts/run_finetune.sh and choose the model you want to use by modify "--model_name_or_path", and also modify the "dataset_path" so that the used dataset is the above dataset. For instance, you can use the model EleutherAI/gpt-neo-1.3B, and set the dataset as:

```sh
dataset_path=${project_dir}/data/hh_rlhf/sft
```

Then, we can run the following command to perform SFT.

```sh
./scripts/run_finetune.sh
```

You can also use lora training by the following command but also need to set the model_name_or_path and dataset appropriately by editing run_finetune_with_lora.sh .

```sh
./scripts/run_finetune_with_lora.sh
```

## Step 2 Reward Modeling

**Data Preparation** Reward modeling involves ranking responses based on the given prompt, typically requiring human labeling. Fortunately, we already have two responses for each prompt in this example, with one preferred over the other. To create a new dataset, we modify the original dataset to include two keys, "positive" and "negative", with the former being the preferred response. Here is an example of the dataset, consisting of two samples.

```sh
{"instances": [
    {"positive": "###Human: Are space heaters safe?###Assistant: What do you mean? Can you give me more details?###Human: I mean, can they be a fire hazard?###Assistant: I don’t know about that.  Can you tell me more about the situation? Are you using the heater on a fireplace?  Do you have any other heater in the room?  What materials are in the room?  Do you feel any heat from the heater?  How are you touching the heater?###Human: Can you just tell me if there are any situations where they could be a fire hazard?###Assistant: The answer is “it depends”.  It would depend on the specific heater you are using, how you are using it, what kind of fuel it uses, what materials you have in the room, and what the surroundings are like.  You’d also have to make sure you’re using it correctly.  Is it switched off?  Is it safe to touch?  It’s not the sort of thing I could just answer by looking at the text you provided.###Human: Thank you. That information is helpful.###Assistant: Glad to be helpful.  I’m always happy to assist.", "negative": "###Human: Are space heaters safe?###Assistant: What do you mean? Can you give me more details?###Human: I mean, can they be a fire hazard?###Assistant: I don’t know about that.  Can you tell me more about the situation? Are you using the heater on a fireplace?  Do you have any other heater in the room?  What materials are in the room?  Do you feel any heat from the heater?  How are you touching the heater?###Human: Can you just tell me if there are any situations where they could be a fire hazard?###Assistant: The answer is “it depends”.  It would depend on the specific heater you are using, how you are using it, what kind of fuel it uses, what materials you have in the room, and what the surroundings are like.  You’d also have to make sure you’re using it correctly.  Is it switched off?  Is it safe to touch?  It’s not the sort of thing I could just answer by looking at the text you provided.###Human: Thank you. That information is helpful.###Assistant: You’re welcome. I’m glad to help. Is there anything else you want to know?"}, 
    {"positive": "###Human: how do i teach my kids to make forts at home?###Assistant: Do you mean you want to teach them to build a simple structure they can hide in? Or something more elaborate?  I don’t know exactly what you mean by a “fort”, but maybe this gives you some ideas?", "negative": "###Human: how do i teach my kids to make forts at home?###Assistant: There are many ways to make forts!  The simplest might be to just pile up all the furniture in one room.  This works if you have a single room in your house, although it might still be a bit challenging to build a taller and sturdier fort this way.  The best way to build forts is to start with something easy, and then have fun exploring how you can improve it."}
]}
```

To start from a model from a previous sft step, you may edit the run_reward_modeling.sh script and update the "dataset_path" to use the desired dataset. Additionally, you can modify the validation_split_percentage parameter to select the last percentage of samples for evaluation. The load_dataset function splits the dataset into training and evaluation sets, which can also be customized by editing the function in /examples/run_reward_modeling.py if you want to prepare your own dataset when running the script.

```python
def build_dataset(tokenizer, config):
    ''' 
    We assume that we have preprocessed the dataset appropriately such that the sample is organized as follows:
    {"positive": prompt + answer_positive, "negative": prompt + answer_negative}, where the positive response is preferred.
    '''
    def tokenize(sample):
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["chosen_input_ids"] = tokenized_pos["input_ids"]
        sample["chosen_attention_mask"] = tokenized_pos["attention_mask"]
        sample["rejected_input_ids"] = tokenized_neg["input_ids"]
        sample["rejected_attention_mask"] = tokenized_neg["attention_mask"]
        return sample

    ds = load_dataset("json", data_files=config.dataset_path, split="train", field="instances")
    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["chosen_input_ids"]) <= 512 and len(x["rejected_input_ids"]) <= 512)
    eval_dataset = None
    if config.validation_split_percentage > 0:
        idx_gap = int((1-config.validation_split_percentage/100) * len(ds))
        train_dataset = ds.select(range(idx_gap))
        eval_dataset = ds.select(range(idx_gap, len(ds)))
    else:
        train_dataset = ds

    return train_dataset, eval_dataset

```

We use the following loss function to train the reward model following the instruct-GPT paper.

```python
    loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

The reward modeling script can be used by 

```sh
./scripts/run_reward_modeling.sh
```

## Examples

We train reward models using the hh-rlhf dataset with three models, LLaMA-7B, GPT-NEO-2.7B, and GPT-NEO-1.3B. The model is first supervised fine-tuned with the training dataset. The reward modeling is trained using the 112K training samples and is evaluated on the 12.5 test samples. 

| Model | Eval Accuracy | Training record |Remarks |
| :----:| :----: | :----: |:----: |
| LLaMA-7B | 79.52% | See  https://wandb.ai/weixiong5237/huggingface/runs/t3uwm8yp | - |
| LLaMA-7B | 71.64% | See  https://wandb.ai/weixiong5237/huggingface/runs/p2ju3r1a | RM from LLaMA without SFT |
| GPT-NEO-2.7B | 69.24% | See https://wandb.ai/weixiong5237/huggingface/runs/8fc1rcf8 | - |
| GPT-NEO-1.3B | 65.58% | See https://wandb.ai/weixiong5237/huggingface/runs/7oemwynu | Only trained on 10000 samples |
