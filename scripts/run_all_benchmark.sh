#!/bin/bash

help_message="./$(basename $0)"
help_message+=" --model_name_or_path MODEL_NAME_OR_PATH"

if [ $# -ge 1 ]; then
  extra_args="$@"
fi

model_name_or_path=""
while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -h|--help)
      printf "${help_message}" 1>&2
      return 0
      ;;
    --model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    *)
      # Ignores unknown options
  esac
  shift
done

model_name=$(echo "${model_name_or_path}" | sed "s/\//--/g")
echo ${model_name}

if [[ "${model_name}" = "" ]]; then
  echo "no model name specified" 1>&2
  exit 1
fi

log_dir=output_dir/${model_name}_lmflow_chat_nll_eval
mkdir -p ${log_dir}
echo "[Evaluating] Evaluate on LMFlow_chat"
./scripts/run_benchmark.sh ${extra_args} --dataset_name lmflow_chat_nll_eval | tee ${log_dir}/benchmark.log 2> ${log_dir}/benchmark.err

log_dir=output_dir/${model_name}_all_nll_eval
mkdir -p ${log_dir}
echo "[Evaluating] Evaluate on [commonsense, wiki, instruction_following (gpt4) ] nll evaluation"
./scripts/run_benchmark.sh ${extra_args} --dataset_name all_nll_eval | tee ${log_dir}/benchmark.log 2> ${log_dir}/benchmark.err

log_dir=output_dir/${model_name}_commonsense_qa_eval
mkdir -p ${log_dir}
echo "[Evaluating] Evaluate on commonsense QA Accuracy evaluation"
./scripts/run_benchmark.sh ${extra_args} --dataset_name commonsense_qa_eval | tee ${log_dir}/benchmark.log 2> ${log_dir}/benchmark.err