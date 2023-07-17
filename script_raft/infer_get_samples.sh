#!/bin/bash
# Please run this script under project directory.

deepspeed_args="--master_port=11110"      # Default argument


exp_id=get_infer_sample
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

mkdir -p ${output_dir} ${log_dir}
#/home/xiongwei/LMFlow/output_models/hh_rlhf_llama_7b_sft_on_relabel_set_blocksize_1024
export PYTHONPATH=.
deepspeed ${deepspeed_args} \
    script_raft/infer_get_samples.py \
    --raft_exp_dir output_models/iter_raft_align \
    --model_name_or_path $1 \
    --raft_mode "raft_get_samples" \
    --raft_random_seed $2 \
    --raft_infer_set $3 \
    --num_raft_iteration 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "constant" \
    --bf16 \
    --deepspeed configs/ds_config_zero2.json \
    --dataset_path ${project_dir}/data/hh_rlhf/rlhf/rlhf_prompt \
    --output_reward_path ${project_dir}/tmp/raft_aligner/reward.txt \
    --output_dir ${output_dir} --overwrite_output_dir \
    --run_name ${exp_id} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 7777 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 12 \
    --inference_batch_size_per_device 8 \
    --collection_strategy "local" \
    --raft_batch_size 2048 \
    --output_min_length 127 \
    --output_max_length 129 \
    --top_reward_percentage 0.125 \
    | tee ${log_dir}/raft_align.log \
    2> ${log_dir}/raft_align.err
