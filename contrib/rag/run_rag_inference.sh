#!/bin/bash
# An interactive inference script without context history, i.e. the chatbot
# won't have conversation memory.

model=gpt2
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

accelerate launch --config_file ../../configs/accelerator_singlegpu_config.yaml \
  rag_inference.py \
    --deepspeed ../../configs/ds_config_chatbot.json \
    --model_name_or_path ${model} \
    --use_accelerator True \
    --max_new_tokens 256 \
    --temperature 1.0 \
    ${lora_args} \
    ${retriever_type} \
    ${corpus_index_path} \
    ${prompt_structure} \
    ${top_k}
