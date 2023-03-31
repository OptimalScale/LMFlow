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

    if [ "$1" = "instruction_ckpt" -o "$1" = "all" ]; then
        echo "downloading instruction_ckpt.tar.gz"
        filename='instruction_ckpt.tar.gz'
        fileid='1d_ioQ-ViVweeifbsFSO4pczc3UORFHZO'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "gpt_ckpt" -o "$1" = "all" ]; then
        echo "downloading gpt_ckpt.tar.gz"
        filename='gpt_ckpt.tar.gz'
        fileid='1JG0ZnF5_HTrrV5liaopb-TBvOjpkmCSo'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"
