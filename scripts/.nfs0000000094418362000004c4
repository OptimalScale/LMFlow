model=Salesforce/blip2-flan-t5-xxl
checkpoint_path=$1
llm_model_name_or_path=$2
deepspeed examples/vis_chatbot_gradio.py --model_name_or_path ${model} \
                                         --deepspeed configs/ds_config_multimodal.json \
                                         --arch_type vision_encoder_decoder \
                                         --task vqa \
                                         --custom_model \
                                         --prompt_format mini_gpt \
                                         --prompt_structure "{input_text}###Assistant:" \
                                         --checkpoint_path ${checkpoint_path} \
                                         --llm_model_name_or_path ${llm_model_name_or_path} \
                                         --low_resource True \
                                         ${@:3}
