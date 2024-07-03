#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4
# Parses arguments
export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name_or_path=gpt2
dataset_path=data/alpaca/train_conversation

# Other optional arguments that can improve memory saving
gradient_checkpointing=True
use_flash_attention=0
gradient_accumulation_steps=1
batch_size=1
block_size=256
per_device_train_batch_size=1
conversation_template=llama2
optimtech=none  # or 'ema' or 'switchema'
optim=dummy
"""
Select an optimizer from the following options:
- 'adamw_hf'
- 'adamw_torch'
- 'adamw_torch_fused'
- 'adamw_torch_xla'
- 'adamw_torch_npu_fused'
- 'adamw_apex_fused'
- 'adafactor'
- 'adamw_anyprecision'
- 'sgd'
- 'adagrad'
- 'adamw_bnb_8bit'
- 'adamw_8bit'
- 'lion_8bit'
- 'lion_32bit'
- 'paged_adamw_32bit'
- 'paged_adamw_8bit'
- 'paged_lion_32bit'
- 'paged_lion_8bit'
- 'rmsprop'
- 'rmsprop_bnb'
- 'rmsprop_bnb_8bit'
- 'rmsprop_bnb_32bit'
- 'galore_adamw'
- 'galore_adamw_8bit'
- 'galore_adafactor'
- 'galore_adamw_layerwise'
- 'galore_adamw_8bit_layerwise'
- 'galore_adafactor_layerwise'
- 'adamp'
- 'sgdp'
- 'adan'
- 'nadam'
- 'radam'
- 'adabound'
- 'adabelief'
- 'adamax'
- 'lamb'
- 'lars'
- 'yogi'
- 'sophia'
- 'adadelta'
- 'adam'
- 'novograd'
- 'adamwschedulefree'
- 'adamwschedulefreeclosure'
- 'sgdschedulefree'
- 'sgdschedulefreeclosure'

"""
learning_rate=1e-5
lr_schedule=cosine
beta1=0.9
beta2=0.999
beta3=0.99
weight_decay=0
momentum=0
num_epoch=0.1
use_deepspeed=1
seed=42

# Safety related arguments
trust_remote_code=0
# EMA and SwitchEMA related arguments
ema_momentum=0
ema_warmup=0
ema_warmup_iters=0
ema_warmup_ratio=0
ema_evaluate_on_ema=False
ema_evaluate_on_nonema=False
ema_full_params_ema=False
ema_update_interval=0

switchema_momentum=0
switchema_warmup=0
switchema_warmup_iters=0
switchema_warmup_ratio=0
switchema_switch_params=0
switchema_switch_by_iter=0
switchema_switch_start=0
switchema_switch_end=0
switchema_switch_interval=0
switchema_full_params_ema=0
switchema_update_interval=0

# Enable model parallelism for multiple gpus, modify this if you prefer
# customized deepspeed zero-redundancy optimization settings
num_gpu=$(python -c "import torch; print(torch.cuda.device_count())")
ds_config_file=configs/ds_config_zero0_no_offload.json
if [[ ${num_gpu} -ge 2 ]]; then
  ds_config_file=configs/ds_config_zero2_no_offload.json
fi

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --lisa_activated_layers)
      lisa_activated_layers="$2"
      shift
      ;;
    --lisa_interval_steps)
      lisa_interval_steps="$2"
      shift
      ;;
    --gradient_checkpointing)
      gradient_checkpointing="$2"
      shift
      ;;
    --deepspeed)
      ds_config_file="$2"
      shift
      ;;
    --use_flash_attention)
      use_flash_attention="$2"
      shift
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      shift
      ;;
    --block_size)
      block_size="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    --per_device_train_batch_size|--batch_size)
      per_device_train_batch_size="$2"
      batch_size="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    --run_name)
      run_name="$2"
      shift
      ;;
    --optim)
      optim="$2"
      shift
      ;;
    --optimtech)
      optimtech="$2"
      shift
      ;;
    --lr)
      learning_rate="$2"
      shift
      ;;
    --beta1)
      beta1="$2"
      shift
      ;;
    --beta2)
      beta2="$2"
      shift
      ;;
    --beta3)
      beta3="$2"
      shift
      ;;
    --weight_decay)
      weight_decay="$2"
      shift
      ;;
    --momentum)
      momentum="$2"
      shift
      ;;
    -n|--num_epoch)
      num_epoch="$2"
      shift
      ;;
    --lr_schedule)
      lr_schedule="$2"
      shift
      ;;
    --use_deepspeed)
      use_deepspeed="$2"
      shift
      ;;
    --seed)
      seed="$2"
      shift
      ;;
    --ema_momentum)
      ema_momentum="$2"
      shift
      ;;
    --ema_warmup)
      ema_warmup="$2"
      shift
      ;;
    --ema_warmup_iters)
      ema_warmup_iters="$2"
      shift
      ;;
    --ema_warmup_ratio)
      ema_warmup_ratio="$2"
      shift
      ;;
    --ema_evaluate_on_ema)
      ema_evaluate_on_ema="$2"
      shift
      ;;
    --ema_evaluate_on_nonema)
      ema_evaluate_on_nonema="$2"
      shift
      ;;
    --ema_full_params_ema)
      ema_full_params_ema="$2"
      shift
      ;;
    --ema_update_interval)
      ema_update_interval="$2"
      shift
      ;;
    --switchema_momentum)
      switchema_momentum="$2"
      shift
      ;;
    --switchema_warmup)
      switchema_warmup="$2"
      shift
      ;;
    --switchema_warmup_iters)
      switchema_warmup_iters="$2"
      shift
      ;;
    --switchema_warmup_ratio)
      switchema_warmup_ratio="$2"
      shift
      ;;
    --switchema_switch_params)
      switchema_switch_params="$2"
      shift
      ;;
    --switchema_switch_by_iter)
      switchema_switch_by_iter="$2"
      shift
      ;;
    --switchema_switch_start)
      switchema_switch_start="$2"
      shift
      ;;
    --switchema_switch_end)
      switchema_switch_end="$2"
      shift
      ;;
    --switchema_switch_interval)
      switchema_switch_interval="$2"
      shift
      ;;
    --switchema_full_params_ema)
      switchema_full_params_ema="$2"
      shift
      ;;
    --switchema_update_interval)
      switchema_update_interval="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

gpu_id=${CUDA_VISIBLE_DEVICES}
deepspeed_args="--master_port=1103${gpu_id::1} --hostfile configs/hostfile --include localhost:${gpu_id}"

optim_suffix_args=""
if [ "${optim}" == "dummy" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_dummy_beta1 ${beta1}"
  optim_suffix_args+=" --optim_dummy_beta2 ${beta2}"
elif [ "${optim}" == "adabelief" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adabelief_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adabelief_beta2 ${beta2}"
  optim_suffix_args+=" --optim_adabelief_weight_decay ${weight_decay}"
elif [ "${optim}" == "adabound" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adabound_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adabound_beta2 ${beta2}"
  optim_suffix_args+=" --optim_adabound_weight_decay ${weight_decay}"
elif [ "${optim}" == "lars" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_lars_momentum ${momentum}"
  optim_suffix_args+=" --optim_lars_weight_decay ${weight_decay}"
elif [ "${optim}" == "lamb" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_lamb_beta1 ${beta1}"
  optim_suffix_args+=" --optim_lamb_beta2 ${beta2}"
  optim_suffix_args+=" --optim_lamb_weight_decay ${weight_decay}"
elif [ "${optim}" == "adamax" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adamax_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adamax_beta2 ${beta2}"
  optim_suffix_args+=" --optim_adamax_weight_decay ${weight_decay}"
elif [ "${optim}" == "nadam" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_nadam_beta1 ${beta1}"
  optim_suffix_args+=" --optim_nadam_beta2 ${beta2}"
  optim_suffix_args+=" --optim_nadam_weight_decay ${weight_decay}"
elif [ "${optim}" == "radam" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_radam_beta1 ${beta1}"
  optim_suffix_args+=" --optim_radam_beta2 ${beta2}"
  optim_suffix_args+=" --optim_radam_weight_decay ${weight_decay}"
elif [ "${optim}" == "adamp" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adamp_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adamp_beta2 ${beta2}"
  optim_suffix_args+=" --optim_adamp_weight_decay ${weight_decay}"
elif [ "${optim}" == "sgdp" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_sgdp_momentum ${momentum}"
  optim_suffix_args+=" --optim_sgdp_weight_decay ${weight_decay}"
elif [ "${optim}" == "yogi" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_yogi_beta1 ${beta1}"
  optim_suffix_args+=" --optim_yogi_beta2 ${beta2}"
  optim_suffix_args+=" --optim_yogi_weight_decay ${weight_decay}"
elif [ "${optim}" == "sophia" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_sophia_beta1 ${beta1}"
  optim_suffix_args+=" --optim_sophia_beta2 ${beta2}"
  optim_suffix_args+=" --optim_sophia_weight_decay ${weight_decay}"
elif [ "${optim}" == "adan" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adan_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adan_beta2 ${beta2}"
  optim_suffix_args+=" --optim_adan_beta3 ${beta3}"
  optim_suffix_args+=" --optim_adan_weight_decay ${weight_decay}"
elif [ "${optim}" == "adam" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adam_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adam_beta2 ${beta2}"
elif [ "${optim}" == "novograd" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_novograd_beta1 ${beta1}"
  optim_suffix_args+=" --optim_novograd_beta2 ${beta2}"
  optim_suffix_args+=" --optim_novograd_weight_decay ${weight_decay}"
elif [ "${optim}" == "adadelta" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
elif [ "${optim}" == "adagrad" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
elif [ "${optim}" == "adamwschedulefree" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_adamwschedulefree_beta1 ${beta1}"
  optim_suffix_args+=" --optim_adamwschedulefree_beta2 ${beta2}"
  optim_suffix_args+=" --optim_adamwschedulefree_weight_decay ${weight_decay}"
elif [ "${optim}" == "sgdschedulefree" ]; then
  optim_suffix_args="--use_customized_optim 1"
  optim_suffix_args+=" --customized_optim ${optim}"
  optim_suffix_args+=" --optim_sgdschedulefree_momentum ${momentum}"
  optim_suffix_args+=" --optim_sgdschedulefree_weight_decay ${weight_decay}"
else
  optim_suffix_args="--optim ${optim}"
  optim_suffix_args+=" --adam_beta1 ${beta1}"
  optim_suffix_args+=" --adam_beta2 ${beta2}"
fi

ema_suffix_args=""
if [ "${optimtech}" == "ema" ]; then
  ema_suffix_args="--use_customized_optimtech 1"
  ema_suffix_args+=" --optimtech_ema_momentum ${ema_momentum}"
  ema_suffix_args+=" --optimtech_ema_warmup ${ema_warmup}"
  ema_suffix_args+=" --optimtech_ema_warmup_iters ${ema_warmup_iters}"
  ema_suffix_args+=" --optimtech_ema_warmup_ratio ${ema_warmup_ratio}"
  ema_suffix_args+=" --optimtech_ema_evaluate_on_ema ${ema_evaluate_on_ema}"
  ema_suffix_args+=" --optimtech_ema_evaluate_on_nonema ${ema_evaluate_on_nonema}"
  ema_suffix_args+=" --optimtech_ema_full_params_ema ${ema_full_params_ema}"
  ema_suffix_args+=" --optimtech_ema_update_interval ${ema_update_interval}"
fi

switchema_suffix_args=""
if [ "${optimtech}" == "switchema" ]; then
  switchema_suffix_args="--use_customized_optimtech 1"
  switchema_suffix_args+=" --optimtech_switchema_momentum ${switchema_momentum}"
  switchema_suffix_args+=" --optimtech_switchema_warmup ${switchema_warmup}"
  switchema_suffix_args+=" --optimtech_switchema_warmup_iters ${switchema_warmup_iters}"
  switchema_suffix_args+=" --optimtech_switchema_warmup_ratio ${switchema_warmup_ratio}"
  switchema_suffix_args+=" --optimtech_switchema_switch_params ${switchema_switch_params}"
  switchema_suffix_args+=" --optimtech_switchema_switch_by_iter ${switchema_switch_by_iter}"
  switchema_suffix_args+=" --optimtech_switchema_switch_start ${switchema_switch_start}"
  switchema_suffix_args+=" --optimtech_switchema_switch_end ${switchema_switch_end}"
  switchema_suffix_args+=" --optimtech_switchema_switch_interval ${switchema_switch_interval}"
  switchema_suffix_args+=" --optimtech_switchema_full_params_ema ${switchema_full_params_ema}"
  switchema_suffix_args+=" --optimtech_switchema_update_interval ${switchema_update_interval}"
fi


# Finetune
exp_id=alpaca_${optim}_lr-${learning_rate}_beta1-${beta1}_beta2-${beta2}_lr-sched-${lr_schedule}_model-$(basename ${model_name_or_path})_batch-size-${batch_size}x${gradient_accumulation_steps}_seed-${seed}
echo "$(date): ${exp_id}..."

tmp_dir=tmp
mkdir -p ${tmp_dir}

prefix=${exp_id}
if [ -f ${tmp_dir}/${prefix}.mark ]; then
  exit 0
fi

trap "rm -f ${tmp_dir}/${prefix}.mark" SIGINT SIGTERM SIGKILL
touch ${tmp_dir}/${prefix}.mark

project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
output_dir=output_models/${exp_id}
mkdir -p ${output_dir} ${log_dir}

exe="deepspeed ${deepspeed_args}"
if [[ ${use_deepspeed} -eq 0 ]]; then
  exe=python
fi
${exe} examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --num_train_epochs ${num_epoch} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type ${lr_schedule} \
    --disable_group_texts 1 \
    --block_size ${block_size} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --bf16 \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --torch_dtype bfloat16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing ${gradient_checkpointing} \
    --use_flash_attention ${use_flash_attention} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --seed ${seed} \
    ${optim_suffix_args} \
    ${ema_suffix_args} \
    ${switchema_suffix_args} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err


if [[ $? -ne 0 ]]; then
  echo "$(date): failed"
  rm -f ${tmp_dir}/${prefix}.mark
fi
