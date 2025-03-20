# Reward Modeling

## Introduction

Reinforcement Learning from Human Feedback (RLHF) requires a reward function to guide the adjustment of the generative model. In this example, we show how to use LMFlow framework to train a reward model following the procedure in the InstructGPT paper: https://arxiv.org/abs/2203.02155 . We use the Dahoas/full-hh-rlhf dataset as an example, where each sample of this dataset consists of a prompt and two responses from the assistant. In particular, the response with label "chosen" is preferred as compared to the response with label "rejected". The dataset consists of 112K training samples and 12.5K test samples. The following is an example sample of the dataset:

Prompt: 

"Human: What kind of noises did dinosaurs make? Assistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made would be Human: yes they did Assistant: to guess, and that would probably require lots of reading and a certain amount of imagination, so we’re not really prepared to do that. Human: you cant read Assistant: 

Chosen response: "You can read?"

Rejected response: "there’s a lot of stuff humans don’t know"

As an example, we prepare 10K sft training samples, 12K reward modeling samples (where 10% of them are split for evaluation) at ./data/hh_rlhf.

## Step 1 Supervised Finetuning (SFT)

We prepare the dataset used for supervised finetuning by adding a prefix to the Human and Assistant inputs to prompt model responses and simplify post-processing. Here is an example of a two-sample dataset to illustrate this.


```json
{
    "type": "conversation",
    "instances": [
        {"messages": [
            {"role": "user", "content": "Instruction: The sentence you are given might be too wordy, complicated, or unclear. Rewrite the sentence and make your writing clearer by keeping it concise. Whenever possible, break complex sentences into multiple sentences and eliminate unnecessary words. If you have any questions about my rate or if you find it necessary to increase or decrease the scope for this project, please let me know."},
            {"role": "assistant", "content": "If you have any questions about my rate or find it necessary to increase or decrease this project's scope, please let me know."}
        ]},
        {"messages": [
            {"role": "user", "content": "Instruction: Rewrite the given text and correct grammar, spelling, and punctuation errors. If you'd told me year ago that today I would finish a marathon, I would of laughed. Your support had a huge affect on me!"},
            {"role": "assistant", "content": "If you'd told me a year ago that today I would finish a marathon, I would have laughed. Your support had a huge effect on me!"}
        ]}
    ]
}
```

See [Finetuning (Full)](./finetuning.md#full-parameters), [Finetuning (LISA)](./finetuning.md#layerwise-importance-sampled-adamw-lisa), and [Finetuning (LoRA)](./finetuning.md#low-rank-adaptation-lora) for more details on the finetuning process.

## Step 2 Reward Modeling

**Data Preparation** Reward modeling involves ranking responses based on the given prompt, typically requiring human labeling. Fortunately, we already have two responses for each prompt in this example, with one preferred over the other. To create a new dataset, we modify the original dataset to include two keys, "positive" and "negative", with the former being the preferred response. Here is an example of the dataset, consisting of two samples.

```json
{"instances": [
    {
        "chosen": [
            {"content": "Please identify the top 5 rarest animals in the world.", "role": "user" }, 
            {"content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant" }, 
            {"content": "The ones that are really rare.", "role": "user" }, 
            {"content": "I’m not sure I can answer this question very easily. I’ll try to give you a basic overview of animals that are rare, and then you can tell me if you think that’s good enough. \n\nIf we were to measure animal populations by the area of land they occupy, the number 1 animal would probably be the Giant Panda. Panda populations are generally estimated to be about 2.1 million Pandas in the wild, which is a pretty small number of animals when we’re thinking about Earth’s entire surface area. The number 2 animal would probably be the Great Auk. Great Auks used to live in North America, and their total population was probably in the low hundreds, but they are now extinct.", "role": "assistant"}
        ],
        "rejected": [
            {"content": "Please identify the top 5 rarest animals in the world.", "role": "user"}, 
            {"content": "Do you mean animals that are really rare, or rare relative to the size of the human population?", "role": "assistant"}, 
            {"content": "The ones that are really rare.", "role": "user"}, 
            {"content": "Alright, here’s what I found:", "role": "assistant"}
        ]
    },
    {
        "chosen": [
            {"content": "How do I teach my kids to make forts at home?", "role": "user"}, 
            {"content": "Do you mean you want to teach them to build a simple structure they can hide in? Or something more elaborate?  I don’t know exactly what you mean by a “fort”, but maybe this gives you some ideas?", "role": "assistant"}
        ],
        "rejected": [
            {"content": "How do I teach my kids to make forts at home?", "role": "user"}, 
            {"content": "There are many ways to make forts!  The simplest might be to just pile up all the furniture in one room.  This works if you have a single room in your house, although it might still be a bit challenging to build a taller and sturdier fort this way.  The best way to build forts is to start with something easy, and then have fun exploring how you can improve it.", "role": "assistant"}
        ]
    }
]
}
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

We train reward models using the hh-rlhf dataset with four models, LLaMA-13B LLaMA-7B, GPT-NEO-2.7B, and GPT-NEO-1.3B. The model is first supervised fine-tuned with the training dataset. The reward modeling is trained using the 112K training samples and is evaluated on the 12.5 test samples. 

The SFT step appears to be crucial, and the number of epochs during SFT can make a difference. The most successful model we obtained was initialized from LLaMA-13B, which underwent SFT on the training dataset for 2 epochs. For reward modeling, we utilize LoRA with a rank of 16. Surprisingly, increasing the LoRA rank to 32 or even 128 does not result in a significant improvement in evaluation accuracy. Moreover, we find that the choice of batch size does not have a significant impact on the training results. Additionally, we observe slight overfitting of the model during the second epoch of reward modeling.

| Model | Eval Accuracy | Training record |Remarks |
| :----:| :----: | :----: |:----: |
| LLaMA-13B | 84.55% | See https://wandb.ai/ianz2020/huggingface/runs/bg677mxa | RM from LLaMA with 2 epochs of SFT |
| LLaMA-13B | 81.80% | See https://wandb.ai/ianz2020/huggingface/runs/ka9v1ywd | RM from LLaMA with 1 epoch of SFT |
| LLaMA-13B | 71.64% | See https://wandb.ai/ianz2020/huggingface/runs/lntwmcyd | RM from LLaMA without SFT |
| LLaMA-7B | 79.52% | See  https://wandb.ai/weixiong5237/huggingface/runs/t3uwm8yp | - |
| LLaMA-7B | 71.64% | See  https://wandb.ai/weixiong5237/huggingface/runs/p2ju3r1a | RM from LLaMA without SFT |
| GPT-NEO-2.7B | 69.24% | See https://wandb.ai/weixiong5237/huggingface/runs/8fc1rcf8 | - |
| GPT-NEO-1.3B | 65.58% | See https://wandb.ai/weixiong5237/huggingface/runs/7oemwynu | Only trained on 10000 samples |
