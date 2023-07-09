model=Salesforce/blip2-flan-t5-xxl
deepspeed examples/debug.py --model_name_or_path ${model} --deepspeed configs/ds_config_multimodal.json --arch_type vision_encoder_decoder --task vqa --custom_model --prompt_format mini_gpt --prompt_structure "{input_text}###Assistant:"
