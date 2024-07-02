model=Salesforce/blip2-opt-2.7b
deepspeed examples/vis_chatbot.py --model_name_or_path ${model} \
                                  --deepspeed configs/ds_config_vis_chatbot.json \
                                  --arch_type vision_encoder_decoder \
                                  --task vqa \
                                  ${@:1}
