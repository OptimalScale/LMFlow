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


## Known Issues
### 1. `undefined symbol` error
When importing the flash attention module, you may encounter `ImportError` saying `undefined symbol`:
```bash
>>> import flash_attn
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File ".../anaconda3/envs/lmflow/lib/python3.9/site-packages/flash_attn/__init__.py", line 3, in <module>
    from flash_attn.flash_attn_interface import flash_attn_func
File ".../anaconda3/envs/lmflow/lib/python3.9/site-packages/flash_attn/flash_attn_interface.py", line 4, in <module>
    import flash_attn_2_cuda as flash_attn_cuda
ImportError: .../anaconda3/envs/lmflow/lib/python3.9/site-packages/flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops9_pad_enum4callERKNS_6TensorEN3c108ArrayRefINS5_6SymIntEEElNS5_8optionalIdEE
```
This MAY due to the incompatibility between the PyTorch version and the flash attention module, or the compiling process of flash attention. We've tested several approaches, either downgrade PyTorch OR upgrade the flash attention module works. If you still encounter this issue, please refer to [this issue](https://github.com/Dao-AILab/flash-attention/issues/451).