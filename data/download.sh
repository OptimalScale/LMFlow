#!/bin/bash

function main() {
    public_server="http://lmflow.org:5000"
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
    
    if [ "$1" = "gpt4_zh_eval" -o "$1" = "all" ]; then
        echo "downloading gpt4_zh_eval dataset"
        filename='gpt4_instruction_zh_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
    
    if [ "$1" = "multiturn_dialog_eval" -o "$1" = "all" ]; then
        echo "downloading multiturn_dialog_eval dataset"
        filename='multiturn_dialog_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
    
    if [ "$1" = "wiki_zh_eval" -o "$1" = "all" ]; then
        echo "downloading wiki_zh_eval dataset"
        filename='wiki_zh_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "wiki_en_eval" -o "$1" = "all" ]; then
        echo "downloading wiki_en_eval dataset"
        filename='wiki_en_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
    
    if [ "$1" = "wiki_en_eval" -o "$1" = "all" ]; then
        echo "downloading wiki_en_eval dataset"
        filename='wiki_en_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
    
    if [ "$1" = "gpt4_en_eval" -o "$1" = "all" ]; then
        echo "downloading gpt4_en_eval dataset"
        filename='gpt4_instruction_en_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
    
    if [ "$1" = "common_sense_eval" -o "$1" = "all" ]; then
        echo "downloading common_sense_eval dataset"
        filename='common_sense_eval.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "hh_rlhf" -o "$1" = "all" ]; then
        echo "downloading hh_rlhf dataset"
        filename='hh_rlhf.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp" -o "$1" = "all" ]; then
        echo "downloading lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp dataset"
        filename='lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "lmflow_chat_cn_dialog_multiturn_single_nll_text2text" -o "$1" = "all" ]; then
        echo "downloading lmflow_chat_cn_dialog_multiturn_single_nll_text2text dataset"
        filename='lmflow_chat_cn_dialog_multiturn_single_nll_text2text.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp" -o "$1" = "all" ]; then
        echo "downloading lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp dataset"
        filename='lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    if [ "$1" = "lmflow_chat_en_dialog_multiturn_single_nll_text2text" -o "$1" = "all" ]; then
        echo "downloading lmflow_chat_en_dialog_multiturn_single_nll_text2text dataset"
        filename='lmflow_chat_en_dialog_multiturn_single_nll_text2text.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

    # multimodal
    if [ "$1" = "coco2017" -o "$1" = "all" ]; then
        echo "downloading coco 2017 dataset for multimodal finetuning"
        mkdir coco2017
        cd coco2017
        wget "http://images.cocodataset.org/zips/train2017.zip"
        wget "http://images.cocodataset.org/zips/val2017.zip"
        wget "http://images.cocodataset.org/zips/test2017.zip"
        unzip train2017.zip
        unzip val2017.zip
        unzip test2017.zip 
        rm train2017.zip
        rm val2017.zip
        rm test2017.zip
        cd ../
    fi

    if [ "$1" = "llava_instruction_finetune_80k" -o "$1" = "all" ]; then
        echo "downloading llava instruction finetune dataset with 80k conversation"
        python ../utils/download_hf_file.py \
            --repo_id liuhaotian/LLaVA-Instruct-150K \
            --filename llava_instruct_80k.json
    fi

    if [ "$1" = "llava_cc3m_pretrain_595k" -o "$1" = "all" ]; then
        echo "downloading llava pretrain images "
        filepath="llava_cc3m_pretrain_595k"
        python ../utils/download_hf_file.py \
            --repo_id liuhaotian/LLaVA-CC3M-Pretrain-595K \
            --filename images.zip \
            --target_path ${filepath}

        python ../utils/download_hf_file.py \
            --repo_id liuhaotian/LLaVA-CC3M-Pretrain-595K \
            --filename chat.json \
            --llava_cc3m_pretrain_595k \
            --target_path ${filepath}

        cd ${filepath}
        unzip images.zip 
        rm -rf images.zip
        cd ../
    fi
}
main "$@"



