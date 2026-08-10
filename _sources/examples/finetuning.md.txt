# Finetuning 

## Full Parameters

Full training updates all the parameters to finetune a language model.
Here is an example to finetune a GPT-2 base model.

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
    --model_name_or_path gpt2 \
    --dataset_path data/alpaca/train_conversation \
    --output_model_path output_models/finetuned_gpt2
```

```{admonition} Conversation Template
:class: tip

For conversation dataset, specify a conversation template for better performance by adding `--conversation_template` to the command.  
```

````{dropdown} Llama-3-8B conversation dataset example
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset_path data/alpaca/train_conversation \
    --conversation_template llama3 \
    --output_model_path output_models/finetuned_llama3_8b
```
````


## Layerwise Importance Sampled AdamW (LISA)

[LISA](https://arxiv.org/abs/2403.17919) is a memory-efficient finetuning algorithm that allows tradeoff between memory and the number of randomly unfreezed layers. This script currently is only tested in single gpus. Please stay tuned for our latest updates!

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lisa.sh \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_path data/alpaca/train_conversation \
    --output_model_path output_models/finetuned_llama2_7b \
    --lisa_activated_layers 1 \
    --lisa_interval_steps 20
```

````{dropdown} Llama-2-7B conversation dataset example
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lisa.sh \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_path data/alpaca/train_conversation \
    --conversation_template llama2 \
    --output_model_path output_models/finetuned_llama2_7b_lisa \
    --lisa_activated_layers 1 \
    --lisa_interval_steps 20
```
````


## Low-Rank Adaptation (LoRA)

LoRA is a parameter-efficient finetuning algorithm and is more efficient than full finetuning.

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
    --model_name_or_path facebook/galactica-1.3b \
    --dataset_path data/alpaca/train_conversation \
    --output_lora_path output_models/finetuned_galactica_lora
```

````{admonition} Merge LoRA Weight
:class: tip

Merge LoRA weight and the base model into one using:  
```sh
./scripts/run_merge_lora.sh \
    --model_name_or_path Qwen/Qwen1.5-1.8B \
    --lora_model_path output_models/lora \
    --output_model_path output_models/lora_merged \
```
````

````{dropdown} Llama-2-7B conversation dataset example
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_path data/alpaca/train_conversation \
    --conversation_template llama2 \
    --output_model_path output_models/finetuned_llama2_7b_lora \
```
````