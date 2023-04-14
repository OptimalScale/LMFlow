#!/bin/bash

model=Tribbiani/vicuna-7b
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed examples/chatbot.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      --max_new_tokens 200 \
      ${lora_args} \
      --prompt_structure "A conversation happens between an intrigued person and a polite artificial intelligence assistant, where the assistant gives helpful, considerate and comprehensive answers to the individual's questions. ###Human: What are the key factors that lead to global warming?.###Assistant: Global warming refers to a long-term increase in the average surface temperature of the earth's atmosphere, primarily caused by greenhouse gas emissions from human activities. Below are some key factors that play a role in global warming:\n\# 1. Burning fossil fuels such as coal, petroleum, and natural gas releases carbon dioxide and other greenhouse gases into the atmosphere. These gases trap heat and contribute to global warming. \n# 2. Deforestation and landfills both contribute to the concentration of carbon dioxide in the atmosphere. Deforestation reduces the number of trees that absorb carbon dioxide, while landfills release methane and carbon dioxide from decomposing waste. \n\# 3. Livestock production, including raising cattle, pigs, and chickens, is a significant source of greenhouse gas emissions. This process releases methane and other gases into the atmosphere and requires large amounts of land and water resources.\n\# 4. Intensive agriculture practices, such as modern technologies and maximizing crop yields, are associated with increased greenhouse gas emissions from livestock manure and fertilizer use. These practices can lead to soil degradation, water pollution, and other environmental problems.\n# 5. Many industries emit large volumes of greenhouse gases during the manufacturing process, including oil and gas production, chemical manufacturing, and steel production. These emissions are often a byproduct of industrial processes, and the significant amount of energy required contributes to greenhouse gas emissions indirectly. Addressing emissions from industry processes is crucial for limiting the impact of climate change.\n###Human: {input_text}###Assistant:" \
      --end_string "###"

