model=Salesforce/blip2-flan-t5-xxl
llm_model_name_or_path=lmsys/vicuna-7b-v1.3
deepspeed_args="--master_port=12000 --num_gpus=1"

if [ ! -f output_models/pretrained_minigpt4_7b.pth ]; then
  cd output_models && ./download.sh minigpt4_7b && cd -
fi

if [ ! -f output_models/pretrained_minigpt4_7b_converted.pth ]; then
  python utils/convert_minigpt4_checkpoints.py \
      --model_path output_models/pretrained_minigpt4_7b.pth \
      --save_path output_models/pretrained_minigpt4_7b_converted.pth
fi

deepspeed ${deepspeed_args} examples/vis_chatbot.py --model_name_or_path ${model} --deepspeed configs/ds_config_vis_chatbot.json --arch_type vision_encoder_decoder --task vqa --custom_model \
                            --chatbot_type mini_gpt \
                            --prompt_structure "{input_text}###Assistant:" \
                            --pretrained_language_projection_path output_models/pretrained_minigpt4_7b_converted.pth \
                            --llm_model_name_or_path ${llm_model_name_or_path} \
                            --low_resource True \
                            ${@:1}

