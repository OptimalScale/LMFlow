# FlashAttention-2
We're thrilled to announce that LMFlow now supports training and inference using **FlashAttention-2**! This cutting-edge feature will take your language modeling to the next level. To use it, simply add ``` --use_flash_attention True ``` to the corresponding bash script.
Here is an example of how to use it:
```
#!/bin/bash
pip install flash_attn==2.0.2

deepspeed --master_port=11000 \
   examples/chatbot.py \                           
      --deepspeed configs/ds_config_chatbot.json \                              
      --model_name_or_path LMFlow/Full-Robin-7b-v2 \                                                     
      --max_new_tokens 1024 \
      --prompt_structure "###Human: {input_text}###Assistant:" \
      --end_string "#" \
      --use_flash_attention True
```

Upgrade to LMFlow now and experience the future of language modeling!
