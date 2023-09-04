model=Salesforce/blip2-flan-t5-xxl
checkpoint_path=/home/qlianab/checkpoints/pretrained_weights/minigpt4/prerained_minigpt4_7b_converted.pth
llm_model_name_or_path=lmsys/vicuna-7b-v1.3
deepspeed_args="--master_port=12000"

deepspeed ${deepspeed_args} examples/vis_chatbot.py --model_name_or_path ${model} --deepspeed configs/ds_config_vis_chatbot.json --arch_type vision_encoder_decoder --task vqa --custom_model \
                            --chatbot_format mini_gpt \
                            --prompt_structure "{input_text}###Assistant:" \
                            --pretrained_language_projection_path ${checkpoint_path} \
                            --llm_model_name_or_path ${llm_model_name_or_path} \
                            --low_resource True \
                            ${@:1}

