#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses argumen
# FIXME upload the projector into lmflow.
model_name_or_path=Salesforce/blip2-flan-t5-xxl
dataset_path=/path/to/llava_instruct_80k.json
image_folder=/path/tococo/train2017
output_dir=output_models/finetune
deepspeed_args="--master_port=12000"

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
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune_multi_modal.py \
    --deepspeed configs/ds_config_multimodal.json \
    --arch_type vision_encoder_decoder \
    --llava_loading True \
    --model_name_or_path ${model_name_or_path} \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --pretrained_language_projection_path /path/to/projector.bin \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --image_folder ${image_folder} \
    --custom_vision_model True \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.5 \
    --image_aspect_ratio None \
    --fp16 True \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --sep_style "v1" \
    --num_train_epochs 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
