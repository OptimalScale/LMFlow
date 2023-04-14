
model_name=${1}
lora_model_name=${2}
exp_id=${3}
log_dir=output_dir/${exp_id}_enppl

if [ -d "${log_dir}" ]; then
	echo "${log_dir} exists"
else
	mkdir -p ${log_dir}
fi

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text \
    --model_name_or_path ${model_name} \
    --lora_model_path ${lora_model_name} \
    --dataset_path data/wikitext-2-raw-v1/test \
    --deepspeed examples/ds_config.json \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
