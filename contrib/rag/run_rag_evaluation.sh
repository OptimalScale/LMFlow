#!/bin/bash

if [ ! -d data/MedQA-USMLE ]; then
  cd data && ./download.sh MedQA-USMLE && cd -
fi

lora_args=""
retriever_type=""
corpus_index_path=""
prompt_structure=""

top_k=""
while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model="$2"
      shift
      ;;
    --lora_model_path)
      lora_args="--lora_model_path $2"
      shift
      ;;
    -r|--retriever_type)
      retriever_type="--retriever_type $2"
      shift
      ;;
    --corpus_index_path)
      corpus_index_path="--corpus_index_path $2"
      shift
      ;;
    --prompt_structure)
      prompt_structure="--prompt_structure $2"
      shift
      ;;
    --top_k_retrieve)
      top_k="--top_k_retrieve $2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/rag_evaluation.py \
  --answer_type medmcqa \
  --model_name_or_path gpt2 \
  --dataset_path data/MedQA-USMLE/validation \
  --deepspeed examples/ds_config.json \
  --inference_batch_size_per_device 1 \
  --metric accuracy
  ${retriever_type} \
  ${corpus_index_path} \
  ${prompt_structure} \
  ${top_k}
