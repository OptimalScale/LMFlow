#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses arguments
model_name_or_path=gpt2
dataset_path=data/alpaca/train
eval_dataset_path=data/alpaca/test
output_dir=output_models/continual-pretrain
num_train_epochs=0.01
per_device_train_batch_size=1
save_steps=1000
deepspeed_args="--master_port=11000"
learning_rate=2e-5
resume_from_checkpoint=None
exp_id=continual-pretrain

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
    -e|--eval_dataset_path)
      eval_dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    -n|--num_train_epochs)
      num_train_epochs=$2
      shift
      ;;
    -b|--per_device_train_batch_size)
      per_device_train_batch_size=$2
      shift
      ;;
    -s|--save_steps)
      save_steps=$2
      shift
      ;;
    -r|--resume_from_checkpoint)
      resume_from_checkpoint=$2
      shift
      ;;
    --lr|--learning_rate)
      learning_rate=$2
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --exp_id)
      exp_id=$2
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

    # --evaluation_strategy steps \
deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${learning_rate} \
    --block_size 2048 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --use_lora 0 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --evaluation_strategy no \
    --eval_steps ${save_steps} \
    --eval_dataset_path ${eval_dataset_path} \
    --ddp_timeout 72000 \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --weight_decay 0 \
    --save_total_limit -1 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
