# only work for gpu mem > 25G; fail to do 4 bit and 8 bit inference.
model_name_or_path=Salesforce/blip2-flan-t5-xxl
llava_pretrain_model_path="output_models/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/"
deepspeed_args="--master_port=12000"

if [ ! -f ${llava_pretrain_model_path}"pytorch_model-00001-of-00003.bin" ]; then
  cd output_models && ./download.sh llava_vicuna13b_model_01 && cd -
fi

if [ ! -f ${llava_pretrain_model_path}"pytorch_model-00002-of-00003.bin" ]; then
  cd output_models && ./download.sh llava_vicuna13b_model_02 && cd -
fi

if [ ! -f ${llava_pretrain_model_path}"pytorch_model-00003-of-00003.bin" ]; then
  cd output_models && ./download.sh llava_vicuna13b_model_03 && cd -
fi


deepspeed ${deepspeed_args} \
    examples/vis_chatbot.py \
    --deepspeed configs/ds_config_vis_chatbot.json \
    --arch_type vision_encoder_decoder \
    --task vqa \
    --custom_model True \
    --chatbot_type llava \
    --prompt_structure '{input_text} ASSISTANT:' \
    --llava_loading True \
    --model_name_or_path ${model_name_or_path} \
    --image_encoder_name_or_path openai/clip-vit-large-patch14-336 \
    --custom_vision_model True \
    --llm_model_name_or_path lmsys/vicuna-13b-v1.5 \
    --llava_pretrain_model_path ${llava_pretrain_model_path}"*.bin" \
    --with_deepspeed False \
    --save_pretrain_model_path "output_models/lmflow_llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3" \
    ${@:1}

