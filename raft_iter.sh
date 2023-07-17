#!/bin/bash

# 定义要运行的次数
count=10

# pinkmanlove/llama-7b-hf
mkdir /home/wxiongae/projects/lmflow_separate_implementation/LMFlow/output_models/0715_iter_raft_align
base_dir="/home/wxiongae/projects/lmflow_separate_implementation/LMFlow/output_models/0715_iter_raft_align"
sft_model="EleutherAI/gpt-neo-2.7B"
reward_model="weqweasdas/hh_rlhf_rm_open_llama_3b"
x=0
y=1
model_dir="${base_dir}/model${x}"
mkdir ${model_dir}
tmp_model_dir="${base_dir}/model${y}"
mkdir $tmp_model_dir
mkdir ${model_dir}/infer_set
mkdir ${model_dir}/filtered_set
mkdir ${tmp_model_dir}/infer_set
mkdir ${tmp_model_dir}/filtered_set


CUDA_VISIBLE_DEVICES="0,2,3" ./script_raft/infer_get_samples.sh $sft_model 1 ${model_dir}/infer_set
CUDA_VISIBLE_DEVICES="0,2,3" ./script_raft/infer_get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${base_dir} ${reward_model}
CUDA_VISIBLE_DEVICES="0,2,3" ./script_raft/sft_finetune.sh $sft_model $tmp_model_dir ${model_dir}/filtered_set


old_model_dir=$tmp_model_dir 

for (( i=2; i<=$count; i++ )); do
  model_dir="${base_dir}/model${i}"
  mkdir $model_dir
  mkdir ${model_dir}/infer_set
  mkdir ${model_dir}/filtered_set
  CUDA_VISIBLE_DEVICES="0,2,3" ./script_raft/infer_get_samples.sh $old_model_dir $((i * 8)) ${old_model_dir}/infer_set
  CUDA_VISIBLE_DEVICES="0,2,3" ./script_raft/infer_get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${base_dir} ${reward_model}
  CUDA_VISIBLE_DEVICES="0,2,3" ./script_raft/sft_finetune.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set

  old_model_dir=$model_dir

  #param=${params[$param_index]}
  #echo "Running script.sh $param"
  #CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" ./scripts_for_uncertainty_study/infer_get_rewards.sh $param/eval_set
  #mkdir 
  #CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" ./scripts_for_uncertainty_study/infer_get_samples.sh $old_model_dir $((i * 8))
  #CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" ./scripts_for_uncertainty_study/infer_get_rewards.sh $param
  #param_index=$(( (param_index + 1) % ${#params[@]} ))
done