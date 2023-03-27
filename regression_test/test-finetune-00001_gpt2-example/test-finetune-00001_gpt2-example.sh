#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

model_name="gpt2"
bash_name="$(basename $0)"
exp_id=${bash_name::-3}

testset_dir=$(pwd)/regression_test/${exp_id}

old_dataset_jsons=$(pwd)/dataset/example_dataset/train.json
new_dataset_dir=${testset_dir}/data/input
mkdir -p ${new_dataset_dir}
cp ${old_dataset_jsons} ${new_dataset_dir}

dataset_name=${exp_id}
if [ ! -f dataset/${dataset_name} ]; then
  ln -s ${new_dataset_dir} dataset/${dataset_name}
fi


output_dir=${testset_dir}/output_models/${exp_id}
log_dir=${testset_dir}/log/${exp_id}

mkdir -p ${output_dir} ${log_dir}

export PYTHONPATH=.
deepspeed --num_gpus=1 --master_port=11000 \
  examples/pytorch/language-modeling/run_clm.py \
    --deepspeed transformers/tests/deepspeed/ds_config_zero2.json \
    --bf16 \
    --run_name ${exp_id} \
    --model_name_or_path ${model_name} \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --dataset_name ${dataset_name} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --output_dir ${output_dir} --overwrite_output_dir \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --is_custom_dataset True \
    --prompt_structure "Defintion: {definition} \n Input: {input} \n Output: {output} \n\n" \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
