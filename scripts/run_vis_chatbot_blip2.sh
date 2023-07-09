model=Salesforce/blip2-opt-2.7b
deepspeed examples/debug.py --model_name_or_path ${model} --deepspeed configs/ds_config_multimodal.json --arch_type vision_encoder_decoder --task vqa
