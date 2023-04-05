#!/bin/bash

function main() {
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) model_name"
        echo "Example: bash $(basename $0) instruction_ckpt"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "llama7b-lora-medical" -o "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-medical.tar.gz"
        filename='llama7b-lora-medical.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-medical" -o "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-medical.tar.gz"
        filename='llama13b-lora-medical.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama30b-lora-medical" -o "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama30b-lora-medical.tar.gz"
        filename='llama30b-lora-medical.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama7b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-170k.tar.gz"
        filename='llama7b-lora-170k.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama7b-lora-380k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-380k.tar.gz"
        filename='llama7b-lora-380k.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-170k.tar.gz"
        filename='llama13b-lora-170k.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-380k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-380k.tar.gz"
        filename='llama13b-lora-380k.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama30b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama30b-lora-170k.tar.gz"
        filename='llama30b-lora-170k.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"
