#!/bin/bash
# accelerate launch --main_process_port 0 ...

# Finetunes
python sft_summarizer.py    \
    --model_name_or_path microsoft/Phi-3-vision-128k-instruct     \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --trust_remote_code \
    --output_dir output_models/finetuned_Phi3 \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --save_strategy "steps" \
    --save_total_limit 2    \
    --lr_scheduler_type "constant" \
    --max_steps -1 \
    --torch_dtype 'bfloat16'    \
    --gradient_checkpointing \
    --logging_strategy  "epoch" \
    --do_eval True \
    --evaluation_strategy 'epoch' \
    --bf16 \
    --bf16_full_eval True \
    --max_seq_length 10000 \
    --attn_implementation 'flash_attention_2' \
    --eval_accumulation_steps 4 \
    --use_peft False\
    --lora_r 16 \
    --lora_alpha 16 \
    --save_only_model True  \
    --overwrite_output_dir True 