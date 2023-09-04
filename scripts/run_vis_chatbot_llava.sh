model=/path/to/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
deepspeed_args="--master_port=12000"

deepspeed ${deepspeed_args} \
    examples/vis_chatbot.py \
    --deepspeed configs/ds_config_vis_chatbot.json \
    --arch_type vision_encoder_decoder \
    --task vqa \
    --custom_model \
    --model_name_or_path ${model} \
    --chatbot_format llava \
    --prompt_structure '{input_text} ASSISTANT:' \
    --low_resource True \
    --llava_loading True \
    --with_deepspeed False \
    ${@:1}

