#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=data4elm/Llama-400M-12L
dataset_path=[PATH TO YOUR DATASET HERE]
# conversation_template=llama2
output_dir=output_models/finetune
deepspeed_args="--master_port=11000"

# Safety related arguments
trust_remote_code=0

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
    -o|--output_dora_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
# Note that project dir will contain files that show you loss and other metrics during finetuning.
exp_id=finetune_with_dora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --block_size 1024 \
    --per_device_train_batch_size 24 \
    --use_dora 1 \
    --lora_r 16 \
    --lora_target_modules="embed_tokens,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head" \
    --save_aggregated_lora 0 \
    --deepspeed configs/ds_config_zero0_no_offload.json \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 128 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
