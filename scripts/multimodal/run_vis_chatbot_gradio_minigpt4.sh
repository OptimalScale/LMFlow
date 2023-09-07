#!/bin/bash

model=Salesforce/blip2-flan-t5-xxl

# if [ ! -f output_models/pretrained_minigpt4_7b.pth ]; then
#   cd output_models && ./download.sh minigpt4_7b && cd -
# fi
# 
# if [ ! -f output_models/pretrained_minigpt4_7b_converted.pth ]; then
#   python utils/convert_minigpt4_checkpoints.py \
#       --model_path output_models/pretrained_minigpt4_7b.pth \
#       --save_path output_models/pretrained_minigpt4_7b_converted.pth
# fi
# 
# deepspeed --master_port=11005 examples/vis_chatbot_gradio.py \
#     --model_name_or_path ${model} \
#     --deepspeed configs/ds_config_multimodal.json \
#     --arch_type vision_encoder_decoder \
#     --task vqa \
#     --custom_model \
#     --chatbot_format mini_gpt \
#     --prompt_structure "###Human: {input_text}###Assistant:" \
#     --llm_model_name_or_path LMFlow/Full-Robin-7b-v2 \
#     --checkpoint_path output_models/pretrained_minigpt4_7b_converted.pth \
#     --low_resource True \
#     --max_new_tokens 1024

if [ ! -f output_models/pretrained_minigpt4_13b.pth ]; then
  cd output_models && ./download.sh minigpt4_13b && cd -
fi

if [ ! -f output_models/pretrained_minigpt4_13b_converted.pth ]; then
  python utils/convert_minigpt4_checkpoints.py \
      --model_path output_models/pretrained_minigpt4_13b.pth \
      --save_path output_models/pretrained_minigpt4_13b_converted.pth
fi

deepspeed --master_port=11005 examples/vis_chatbot_gradio.py \
    --model_name_or_path ${model} \
    --deepspeed configs/ds_config_vis_chatbot.json \
    --arch_type vision_encoder_decoder \
    --task vqa \
    --custom_model \
    --chatbot_type mini_gpt \
    --prompt_structure "###Human: {input_text}###Assistant:" \
    --llm_model_name_or_path LMFlow/Full-Robin-13b-v2 \
    --pretrained_language_projection_path output_models/pretrained_minigpt4_13b_converted.pth \
    --low_resource True \
    --max_new_tokens 1024
