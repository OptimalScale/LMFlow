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
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "MedQA-USMLE" -o "$1" = "all" ]; then
        echo "downloading MedQA-USMLE"
        filename='MedQA-USMLE.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "ni" ]; then
        echo "downloading natural-instructions"
        filename='natural-instructions.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "PubMedQA" -o "$1" = "all" ]; then
        echo "downloading PubMedQA"
        filename='PubMedQA.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "example_dataset" -o "$1" = "all" ]; then
        echo "downloading example_dataset"
        filename='example_dataset.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "alpaca" -o "$1" = "all" ]; then
        echo "downloading alpaca dataset"
        filename='alpaca.tar.gz'
        wget 144.214.54.164:5000/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"
