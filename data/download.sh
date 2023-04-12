#!/bin/bash

function main() {
    public_server="http://18.207.214.227:5000"
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) dataset_name"
        echo "Example: bash $(basename $0) MedMCQA"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "MedMCQA" -o "$1" = "all" ]; then
        echo "downloading MedMCQA"
        filename='MedMCQA.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "MedQA-USMLE" -o "$1" = "all" ]; then
        echo "downloading MedQA-USMLE"
        filename='MedQA-USMLE.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "ni" ]; then
        echo "downloading natural-instructions"
        filename='natural-instructions.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "PubMedQA" -o "$1" = "all" ]; then
        echo "downloading PubMedQA"
        filename='PubMedQA.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "example_dataset" -o "$1" = "all" ]; then
        echo "downloading example_dataset"
        filename='example_dataset.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "alpaca" -o "$1" = "all" ]; then
        echo "downloading alpaca dataset"
        filename='alpaca.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "red_teaming" -o "$1" = "all" ]; then
        echo "downloading red_teaming dataset"
        filename='red_teaming.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "wikitext-2-raw-v1" -o "$1" = "all" ]; then
        echo "downloading wikitext-2-raw-v1 dataset"
        filename='wikitext-2-raw-v1.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "imdb" -o "$1" = "all" ]; then
        echo "downloading imdb dataset"
        filename='imdb.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "wiki_cn" -o "$1" = "all" ]; then
        echo "downloading wiki_cn dataset"
        filename='wiki_cn.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"
