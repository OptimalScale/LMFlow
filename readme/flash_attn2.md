# Flash Attention 2.0
We're thrilled to announce that LMFlow now supports training and inference using **FlashAttention-2**! This cutting-edge feature will take your language modeling to the next level. To use it, simply add ``` --use_flash_attention True ``` to the corresponding bash script.
Here is an example of how to use it:
```
#!/bin/bash
pip install flash_attn==2.0.2

model=pinkmanlove/llama-7b-hf
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/chatbot.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      --use_flash_attention True \
      ${lora_args}
```

Upgrade to LMFlow now and experience the future of language modeling!