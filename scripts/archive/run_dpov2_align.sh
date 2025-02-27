#!/bin/bash

# Parses arguments
run_name=dpov2_align
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
reference_model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
dataset_path=data/iterative-prompt/train
eval_dataset_path=data/iterative-prompt/eval
output_dir=output_models/${run_name}

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -r|--run_name)
      run_name="$2"
      shift
      ;;
    --model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    --reference_model_name_or_path)
      reference_model_name_or_path="$2"
      shift
      ;;
    --dataset_path)
      dataset_path="$2"
      shift
      ;;
    --eval_dataset_path)
      eval_dataset_path="$2"
      shift
      ;;
    -o|--output_dir)
      output_dir="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${run_name}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_dsz3_config.yaml \
  examples/dpov2_train.py \
    --model_name_or_path ${model_name_or_path} \
    --reference_model_name_or_path ${reference_model_name_or_path} \
    --do_train True \
    --dataset_path ${dataset_path} \
    --eval_dataset_path ${eval_dataset_path} \
    --bf16 True \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --optim paged_adamw_32bit \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --margin_scale 1.0 \
    --max_prompt_length 1000 \
    --num_train_epochs 2 \
    --logging_steps 2 \
    --save_strategy epoch \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --loss_type sigmoid \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --sampling_paired_method max_min \
    --report_to wandb \
    --mask_prompt True \
    --length_penalty 0 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err