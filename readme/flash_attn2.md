# Flash Attention 2.0
We're thrilled to announce that LMFlow now supports training and inference using **FlashAttention-2**! This cutting-edge feature will take your language modeling to the next level. To use it, simply add ``` --use_flash_attention True ``` to the corresponding bash script.
Here is an example of how to use it:
```
#!/bin/bash

deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path pinkmanlove/llama-7b-hf \
    --dataset_path data/wiki_en_eval \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --block_size 2048 \
    --use_flash_attention True \
    --metric ppl
```
Upgrade to LMFlow now and experience the future of language modeling!