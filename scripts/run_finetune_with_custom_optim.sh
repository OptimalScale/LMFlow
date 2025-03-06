#!/bin/bash
model_name_or_path=meta-llama/Llama-3.2-3B-Instruct
dataset_path=data/alpaca/train_conversation
conversation_template=llama3
output_dir=output_models/finetune_custom_optim

# Select an optimizer from the following options:
# - 'adamw_torch'
# - 'adafactor'
# - 'sgd'
# - 'lion_8bit'
# - 'lion_32bit'
# - 'rmsprop'
# Additional optimizers are shown below
optim=dummy
beta1=0.9
beta2=0.999
beta3=0.99
weight_decay=0
momentum=0

# Finetune
exp_id=finetune_custom_optim
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

optim_suffix_args=""
if [ "${optim}" == "dummy" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_dummy_beta1 ${beta1}"
  optim_suffix_args+=" --optim_dummy_beta2 ${beta2}"
elif [ "${optim}" == "adabelief" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "adabound" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "lars" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_momentum ${momentum}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "lamb" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "adamax" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "nadam" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "radam" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "adamp" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "sgdp" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_momentum ${momentum}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "yogi" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "sophia" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "adan" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_beta3 ${beta3}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "adam" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
elif [ "${optim}" == "novograd" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "adadelta" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
elif [ "${optim}" == "adagrad" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
elif [ "${optim}" == "adamw_schedule_free" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_beta1 ${beta1}"
  optim_suffix_args+=" --optim_beta2 ${beta2}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
elif [ "${optim}" == "sgd_schedule_free" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_momentum ${momentum}"
  optim_suffix_args+=" --optim_weight_decay ${weight_decay}"
else
  optim_suffix_args="--optim ${optim}"
  optim_suffix_args+=" --adam_beta1 ${beta1}"
  optim_suffix_args+=" --adam_beta2 ${beta2}"
fi

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code 0 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    ${optim_suffix_args} \
    --disable_group_texts 1 \
    --num_train_epochs 1 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 \
    --torch_dtype bfloat16 \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --use_flash_attention 0 \
    --gradient_checkpointing 0 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --run_name ${exp_id} \
    --seed 42 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)
