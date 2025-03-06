#!/bin/bash
# Parses arguments
model_name_or_path=meta-llama/Llama-3.2-3B-Instruct
dataset_path=data/alpaca/train_conversation
conversation_template=llama3
output_dir=output_models/finetune_qlora

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
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
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
exp_id=finetune_with_lora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --conversation_template ${conversation_template} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 0.01 \
    --learning_rate 2e-5 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --use_qlora 1 \
    --bits 4 \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --run_name ${exp_id} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err