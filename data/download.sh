#!/bin/bash

function main() {
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) dataset_name"
        echo "Example: bash $(basename $0) MedMCQA"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "MedMCQA" -o "$1" = "all" ]; then
        echo "downloading MedMCQA"
        filename='MedMCQA.tar.gz'
        fileid='1xrjtWYmPB5o-dx2v03atYCaehVb5h7xc'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "MedQA-USMLE" -o "$1" = "all" ]; then
        echo "downloading MedQA-USMLE"
        filename='MedQA-USMLE.tar.gz'
        fileid='162zJOHnIpNHlRgssv0aww0a3ABOaV_w2'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "ni"]; then
        echo "downloading natural-instructions"
        filename='natural-instructions.tar.gz'
        fileid='1uq62MnN3V2gV1pWtrl8TBIEzxgeX-AGR'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "PubMedQA" -o "$1" = "all" ]; then
        echo "downloading PubMedQA"
        filename='PubMedQA.tar.gz'
        fileid='1O0MpwkCwTpsV5E_6AWg4RvpHEQYtTIwq'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "example_dataset" -o "$1" = "all" ]; then
        echo "downloading example_dataset"
        filename='example_dataset.tar.gz'
        fileid='1Y1Uzj3e2dXv8GO_KG60N3nUVv2L9Yd0p'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "alpaca" -o "$1" = "all" ]; then
        echo "downloading alpaca dataset"
        filename='alpaca.tar.gz'
        fileid='18R_IDLtWm0ZlMd4R1WBMVJA-K9-kvici'
        wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"
