#!/bin/bash
# Please run this script under project directory.

deepspeed_args="--master_port=11112"      # Default argument


exp_id=test_infer_reward
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

mkdir -p ${output_dir} ${log_dir}
#/home/xiongwei/LMFlow/output_models/relabel_infer/all_gen_results
export PYTHONPATH=.
deepspeed ${deepspeed_args} \
    script_raft/infer_get_rewards.py \
    --model_name_or_path gpt2\
    --raft_exp_dir $3 \
    --reward_model_or_path $4 \
    --raft_mode "raft_get_rewards" \
    --num_raft_iteration 999 \
    --learning_rate 2e-5 \
    --raft_infer_set $1 \
    --raft_filtered_set $2 \
    --lr_scheduler_type "constant" \
    --bf16 \
    --deepspeed configs/ds_config_zero2.json \
    --dataset_path $1 \
    --output_reward_path ${project_dir}/tmp/raft_aligner/reward.txt \
    --output_dir ${output_dir} --overwrite_output_dir \
    --run_name ${exp_id} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 7777 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 12 \
    --inference_batch_size_per_device 2 \
    --collection_strategy "top" \
    --raft_batch_size 1024 \
    --output_min_length 128 \
    --output_max_length 196 \
    --top_reward_percentage 0.125 \
    | tee ${log_dir}/raft_align.log \
    2> ${log_dir}/raft_align.err
