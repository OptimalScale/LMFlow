model=Salesforce/blip2-flan-t5-xxl
checkpoint_path=/scratch/PI/tongzhang/qinglian/checkpoints/pretrained_weights/minigpt4/prerained_minigpt4_7b_converted.pth
llm_model_name_or_path=/scratch/PI/tongzhang/qinglian/checkpoints/pretrained_weights/vicuna-7b/
deepspeed examples/vis_chatbot.py --model_name_or_path ${model} --deepspeed configs/ds_config_multimodal.json --arch_type vision_encoder_decoder --task vqa --custom_model \
                            --prompt_format mini_gpt \
                            --prompt_structure "{input_text}###Assistant:" \
                            --checkpoint_path ${checkpoint_path} \
                            --llm_model_name_or_path ${llm_model_name_or_path} \
                            --low_resource True \
                            ${@:1}

