
exp_id=${1}
dataset_name=${2}
model_name=${3}
lora_model_name=${4}
log_dir=output_dir/${exp_id}_nll

if [ -d "${log_dir}" ]; then
	echo "${log_dir} exists"
else
	mkdir -p ${log_dir}
fi

lora_args=""
if [ $# -ge 3 ]; then
  model=$1
fi
if [ $# -ge 4 ]; then
  lora_args="--lora_model_path $3"
fi

if [ $# -ge 5 ]; then
  deepspeed_args="$5"
fi


CUDA_VISIBLE_DEVICES=0 \
    deepspeed $deepspeed_args examples/benchmarking.py \
    --answer_type text2text \
    --use_ram_optimized_load 0 \
    --model_name_or_path ${model_name} \
    ${lora_args} \
    --dataset_name ${dataset_name}\
    --deepspeed examples/ds_config.json \
    --metric nll \
    --prompt_structure "###Human: {input}###Assistant:" \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err 

