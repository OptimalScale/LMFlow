#!/bin/bash

model=gpt2-xl
gpu=3
gamma=5
max_new_tokens=50
temperature=0
interactive=0

while true; do
  user_input=$(tr -dc A-Za-z0-9 </dev/urandom  | head -c 10)
  echo "$(date): test user_input: \"${user_input}\""

  file_list=""
  for draft_model in gpt2 gpt2-medium; do
    log_file=log/${model}_with_${draft_model}.log
    output_file=log/output_${model}_with_${draft_model}.log
  
    python ./examples/speculative_inference.py \
      --gpu ${gpu} \
      --model ${model} \
      --draft_model ${draft_model} \
      --gamma ${gamma} \
      --max_new_tokens ${max_new_tokens} \
      --temperature ${temperature} \
      --interactive ${interactive} \
      --user_input ${user_input} \
      > ${log_file}
    cat ${log_file} | grep -v WARNING | grep -v INFO | tee ${output_file}
    file_list+=" ${output_file}"
  done
  diff ${file_list}
  if [ $? -ne 0 ]; then
    echo "find diff!!!"
    echo "user input: ${user_input}"
    break
  fi
done
