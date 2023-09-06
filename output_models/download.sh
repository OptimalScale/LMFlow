#!/bin/bash

function main() {
    public_server="http://lmflow.org:5000"
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) model_name"
        echo "Example: bash $(basename $0) instruction_ckpt"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "llama7b-lora-medical" -o "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-medical.tar.gz"
        filename='llama7b-lora-medical.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-medical" -o "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-medical.tar.gz"
        filename='llama13b-lora-medical.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama30b-lora-medical" -o "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama30b-lora-medical.tar.gz"
        filename='llama30b-lora-medical.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama7b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-170k.tar.gz"
        filename='llama7b-lora-170k.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama7b-lora-380k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-380k.tar.gz"
        filename='llama7b-lora-380k.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-170k.tar.gz"
        filename='llama13b-lora-170k.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-380k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-380k.tar.gz"
        filename='llama13b-lora-380k.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama30b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama30b-lora-170k.tar.gz"
        filename='llama30b-lora-170k.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama7b-lora-movie-reviewer" -o "$1" = "raft_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-movie-reviewer"
        filename='llama7b-lora-movie-reviewer.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "cockatoo-7b" -o "$1" = "all" ]; then
        echo "downloading cockatoo-7b"
        filename='cockatoo-7b.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "parakeets-2.7b" -o "$1" = "all" ]; then
        echo "downloading parakeets-2.7b"
        filename='parakeets-2.7b.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "robin-7b" -o "$1" = "all" ]; then
        echo "downloading robin-7b"
        filename='robin-7b-v2-delta.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "minigpt4_7b" -o "$1" = "all" ]; then
        echo "downloading minigpt4_7b"
        filename='pretrained_minigpt4_7b.pth'
        wget ${public_server}/${filename}
    fi

    if [ "$1" = "minigpt4_13b" -o "$1" = "all" ]; then
        echo "downloading minigpt4_13b"
        filename='pretrained_minigpt4_13b.pth'
        wget ${public_server}/${filename}
    fi
    
    if [ "$1" = "llava_vicuna7b_language_projection" -o "$1" = "all"  ]; then
        echo "downloading llava vicuna7b language_projection"
        filename="multimodal/llava-336px-pretrain-vicuna-7b-v1.3_language_projection.pth"
        wget ${public_server}/${filename}
    fi

    if [ "$1" = "llava_vicuna13b_model_01" -o "$1" = "all" ]; then
        echo "downloading llava vicunda13 b trained model 01 checkpoint "
        filepath="llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
        filename="pytorch_model-00001-of-00003.bin"
        python ../utils/download_hf_file.py \
            --repo_id liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
            --filename ${filename} \
            --target_path ${filepath} \
            --repo_type "model"
    fi

    if [ "$1" = "llava_vicuna13b_model_02" -o "$1" = "all" ]; then
        echo "downloading llava vicunda13 b trained model 01 checkpoint "
        filepath="llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
        filename="pytorch_model-00002-of-00003.bin"
        python ../utils/download_hf_file.py \
            --repo_id liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
            --filename ${filename} \
            --target_path ${filepath} \
            --repo_type "model"
    fi

    if [ "$1" = "llava_vicuna13b_model_03" -o "$1" = "all" ]; then
        echo "downloading llava vicunda13 b trained model 01 checkpoint "
        filepath="llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
        filename="pytorch_model-00003-of-00003.bin"
        python ../utils/download_hf_file.py \
            --repo_id liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
            --filename ${filename} \
            --target_path ${filepath} \
            --repo_type "model"
    fi
}

main "$@"
