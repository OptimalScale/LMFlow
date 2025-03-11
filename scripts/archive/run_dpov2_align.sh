#!/bin/bash
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
dataset_path=data/iterative-prompt/train
output_dir=output_models/dpov2_align

# DPO related arguments
reference_model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
eval_dataset_path=data/iterative-prompt/eval
margin_scale=1.0
max_prompt_length=1000
loss_type=sigmoid
sampling_paired_method=max_min
mask_prompt=True
length_penalty=0

# Align
exp_id=dpov2_align
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
  examples/dpov2_train.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code 0 \
    --reference_model_name_or_path ${reference_model_name_or_path} \
    --do_train True \
    --dataset_path ${dataset_path} \
    --eval_dataset_path ${eval_dataset_path} \
    --margin_scale ${margin_scale} \
    --max_prompt_length ${max_prompt_length} \
    --loss_type ${loss_type} \
    --sampling_paired_method ${sampling_paired_method} \
    --mask_prompt ${mask_prompt} \
    --length_penalty ${length_penalty} \
    --bf16 True \
    --torch_dtype bfloat16 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --optim paged_adamw_32bit \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --num_train_epochs 2 \
    --logging_steps 2 \
    --save_strategy epoch \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --output_dir ${output_dir} \
    --run_name ${exp_id} \
    --report_to wandb \
    --seed 42 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err