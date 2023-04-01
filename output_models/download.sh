#!/bin/bash

function main() {
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) model_name"
        echo "Example: bash $(basename $0) instruction_ckpt"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "medical_ckpt" -o "$1" = "all" ]; then
        echo "downloading medical_ckpt.tar.gz"
        filename='medical_ckpt.tar.gz'
        fileid='1bnsQGNGNYchsOfiNyRAmL2fNiowbmFNw'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "gpt_neo2.7B_full_170k" -o "$1" = "gpt_ckpt" -o "$1" = "all" ]; then
        echo "downloading gpt_neo2.7B_full_170k.tar.gz"
        filename='gpt_neo2.7B_full_170k.tar.gz'
        fileid='12Oxj6WgaLNUokmEkdVcOhf-OaEUNjJNl'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "gpt2_full_170k" -o "$1" = "gpt_ckpt" -o "$1" = "all" ]; then
        echo "downloading gpt2_full_170k.tar.gz"
        filename='gpt2_full_170k.tar.gz'
        fileid='14ppgMWYKHcCetlkBXWkVhq_8a6ScB14d'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "gpt2_large_full_170k" -o "$1" = "gpt_ckpt" -o "$1" = "all" ]; then
        echo "downloading gpt2_large_full_170k.tar.gz"
        filename='gpt2_large_full_170k.tar.gz'
        fileid='1tY4cr_01TFiEau7cq52pzEVAGstuHLVH'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama7b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama7b-lora-170k.tar.gz"
        filename='llama7b-lora-170k.tar.gz'
        fileid='1JEeO5tUb-hR9l4cPeNVbiTcdYYjOIAib'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama13b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama13b-lora-170k.tar.gz"
        filename='llama13b-lora-170k.tar.gz'
        fileid='1M1fS9N0OxqoNvzn9J9baooQm0TWVXBh6'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "llama30b-lora-170k" -o "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading llama30b-lora-170k.tar.gz"
        filename='llama30b-lora-170k.tar.gz'
        fileid='1IqgqLHwNkWQ7BffheZnqD6a-8Zul1bk6'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"
