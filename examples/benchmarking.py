#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import json
import logging
import os 
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

import subprocess

from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments, BenchmarkingArguments

logger = logging.getLogger(__name__)


LOCAL_DATSET_MAP ={
    "gpt4_en_eval":"data/gp4_instruction_en_eval/",
    "gpt4_zh_eval":"data/gpt4_instruction_zh_eval/",
    "wiki_zh_eval":"data/wiki_zh_eval/",
    "wiki_en_eval":"data/wiki_en_eval/",
    "multiturn_dialog_eval":"data/multiturn_dialog_eval/",
    "MedMCQA":"data/MedMCQA/validation/",
    "MedQA-USMLE":"data/MedQA-USMLE/validation/",
    "PubMedQA":"data/PubMedQA/test/",
    "alpaca":"data/alpaca/test/",
    "common_sense_eval_arc_c":"data/commonsense_eval/arc_c",
    "common_sense_eval_arc_e":"data/commonsense_eval/arc_e",
    "common_sense_eval_winogrande":"data/commonsense_eval/winogrande/",
    "common_sense_eval_obqa":"data/commonsense_eval/obqa/",
    "common_sense_eval_piqa":"data/commonsense_eval/piqa/",
    "common_sense_eval_hellaswag":"data/commonsense_eval/hellaswag/",
    "common_sense_eval_siqa":"data/commonsense_eval/siqa/",
    "common_sense_eval_boolq":"data/commonsense_eval/boolq/",
    "lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp":"data/lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp",
    "lmflow_chat_cn_dialog_multiturn_single_nll_text2text":"data/lmflow_chat_cn_dialog_multiturn_single_nll_text2text",
    "lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp":"data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp",
    "lmflow_chat_en_dialog_multiturn_single_nll_text2text":"data/lmflow_chat_en_dialog_multiturn_single_nll_text2text",
}

LM_EVAL_DATASET_MAP={
    "commonsense_qa_eval":"openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq",
    "math_eval":"gsm8k",
    'boolq':"boolq",
}


LOCAL_DATSET_GROUP_MAP={
    "commonsense_nll_eval":"common_sense_eval_arc_c,common_sense_eval_arc_e,common_sense_eval_winogrande,\
    common_sense_eval_obqa,common_sense_eval_piqa,common_sense_eval_hellaswag,common_sense_eval_siqa,\
    common_sense_eval_boolq",
    "gpt4_en_eval":"gpt4_en_eval",
    "gpt4_zh_eval":"gpt4_zh_eval",
    "wiki_zh_eval":"wiki_zh_eval",
    "wiki_en_eval":"wiki_en_eval",
    "wiki_eval":"wiki_zh_eval,wiki_en_eval",
    "multiturn_dialog_eval":"multiturn_dialog_eval",
    "all_nll_eval":"common_sense_eval_arc_c,common_sense_eval_arc_e,common_sense_eval_winogrande,\
    common_sense_eval_obqa,common_sense_eval_piqa,common_sense_eval_hellaswag,common_sense_eval_siqa,\
    common_sense_eval_boolq,gpt4_en_eval,gpt4_zh_eval,wiki_zh_eval,wiki_en_eval,\
    lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp,lmflow_chat_cn_dialog_multiturn_single_nll_text2text,\
    lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp,lmflow_chat_en_dialog_multiturn_single_nll_text2text",
    "lmflow_chat_nll_eval":"lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp,lmflow_chat_cn_dialog_multiturn_single_nll_text2text,\
    lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp,lmflow_chat_en_dialog_multiturn_single_nll_text2text",
    "lmflow_chat_zh_nll_eval":"lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp,lmflow_chat_cn_dialog_multiturn_single_nll_text2text",
}

LOCAL_DATSET_ANSWERTYPE_MAP={
    "gpt4_en_eval":"text2text",
    "gpt4_zh_eval":"text2text",
    "wiki_zh_eval":"text_only",
    "wiki_en_eval":"text_only",
    "multiturn_dialog_eval":"text2text",
    "MedMCQA":"multiple_choice",
    "MedQA-USMLE":"multiple_choice",
    "PubMedQA":"binary_choice",
    "alpaca":"text_only",
    "common_sense_eval_arc_c":"text_only",
    "common_sense_eval_arc_e":"text_only",
    "common_sense_eval_winogrande":"text_only",
    "common_sense_eval_obqa":"text_only",
    "common_sense_eval_piqa":"text_only",
    "common_sense_eval_hellaswag":"text_only",
    "common_sense_eval_siqa":"text_only",
    "common_sense_eval_boolq":"text_only",
    "lmflow_chat_cn_dialog_multiturn_nll_text2text_nosharp":"text2text",
    "lmflow_chat_cn_dialog_multiturn_single_nll_text2text":"text2text",
    "lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp":"text2text",
    "lmflow_chat_en_dialog_multiturn_single_nll_text2text":"text2text",
}

    
def is_lmflow_local_benchmarking(dataset_name):
    # local_dataset = ["gpt4_en_eval","gpt4_zh_eval","wiki_zh_eval",\
    #     "multiturn_dialog_eval","MedMCQA","MedQA-USMLE","PubMedQA",\
    #     "alpaca","common_sense_eval_arc_c","common_sense_eval_arc_e",\
    #     "common_sense_eval_winogrande","common_sense_eval_obqa",\
    #     "common_sense_eval_piqa","common_sense_eval_hellaswag",\
    #     "common_sense_eval_siqa","common_sense_eval_boolq"]
    if dataset_name in LOCAL_DATSET_GROUP_MAP.keys():
        dataset_name_collection = LOCAL_DATSET_GROUP_MAP[dataset_name]
        for dataset_name_exact in dataset_name_collection.split(','):
            dataset_name_exact = dataset_name_exact.strip()
            if "common_sense_eval" in dataset_name_exact:
                dataset_list = dataset_name_exact.split(",")
                for common_exact_ in dataset_list:
                    print("Dealing with "+common_exact_.strip())
                    common_exact_ = common_exact_.strip()
                    if not os.path.exists(LOCAL_DATSET_MAP[common_exact_]):
                        os.system("cd data && ./download.sh common_sense_eval && cd -")
            else:
                if not os.path.exists(LOCAL_DATSET_MAP[dataset_name_exact]):
                    print("Checking if dataset "+ str(dataset_name_exact) + " exists")
                    os.system("cd data && "+'./download.sh '+dataset_name_exact +" && cd -")
        return True
    else:
        return False

def is_lm_evaluation_benchmarking(dataset_name):
    if dataset_name in LM_EVAL_DATASET_MAP.keys():
        return True
    else:
        return False

def run_lmflow_local_benchmarking(dataset_name,pipeline_name,model_args, \
    pipeline_args, model, local_metric="neg_log_likelihood"):
    # Downloads dataset via "data/download.sh"
    print('dataset_name.split')
    print(dataset_name.split(","))
    result_list = []
    dataset_name = LOCAL_DATSET_GROUP_MAP[dataset_name]
    dataset_collection = dataset_name.split(",")
    reuslt_collection = []
    for dataset_name_ in dataset_collection:
        # Gets mapping from dataset_name to dataset 
        dataset_name_ = dataset_name_.strip()
        dataset_path = LOCAL_DATSET_MAP[dataset_name_]
        data_args = DatasetArguments(dataset_path=dataset_path)
        dataset = Dataset(data_args)
        
        logger.warning("Default answer type for lmflow local benchmark tasks. \
                       Users need to change answer type in LOCAL_DATSET_ANSWERTYPE_MAP for new benchmark tasks.")
        pipeline_args.answer_type = LOCAL_DATSET_ANSWERTYPE_MAP[dataset_name_]
        
        evaluator = AutoPipeline.get_pipeline(
            pipeline_name=pipeline_name,
            model_args=model_args,
            data_args=data_args,
            pipeline_args=pipeline_args,
        )
        # model = model_args.model_name_or_path
        # metric should be decided by both dataset_name and pipeline_args 
        # 1. When --metric is not specified, or "accuracy", log warning and change to
        #    the dataset_name's default metric
        # 2. If specified, use the specified metric
        result = evaluator.evaluate(model=model, dataset=dataset, metric=local_metric,verbose=True)
        reuslt_collection.append({"dataset":dataset_name_,"result":result})
    for record in reuslt_collection:
        print("-"*30)
        print("| Dataset: " + record['dataset'] )
        print("-"*30)
        print(f"| current nll: {record['result']}")
        print("-"*30)


def run_lm_evaluation_benchmarking(dataset_name,model_name):
    # use subprocess maybe
    # subprocess.run(["python3", "main.py", "--model", "hf-causal-experimental", 
    # "--model_args" "pretrained=EleutherAI/gpt-j-6b",
    # "--tasks", "openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq"
    # "--device", "cuda:0"])
    dataset = LM_EVAL_DATASET_MAP[dataset_name]
    subprocess.run(["python3", "utils/lm_evaluator.py", "--model", "hf-causal-experimental", 
    "--model_args", "pretrained="+model_name,
    "--tasks", dataset,
    "--device", "cuda:0"])

def main():
    # Parses arguments (self-defined for our evaluation platform)
    # Need at least include
    #   1) model_name_or_path (you can reuse ModelArguments)
    #   2) dataset_name
    #   3) metric (you can reuse EvaluatorArguments)
    logging.basicConfig()
    pipeline_name = "evaluator"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments, PipelineArguments, BenchmarkingArguments
    ))
    model_args, pipeline_args, benchmarking_args = parser.parse_args_into_dataclasses()

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)
    # Based on dataset name, you need specify the default dataset_path
    # (if local) or call corresponding lm_evaluation package (via python pack or subprocess)
    dataset_name = benchmarking_args.dataset_name
    # metric = pipeline_args.metric
    if is_lmflow_local_benchmarking(dataset_name):   # TODO (@Jipeng)
        model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)
        run_lmflow_local_benchmarking(dataset_name,pipeline_name,model_args,pipeline_args,model)  # Pass args TODO (@Jipeng)
    elif is_lm_evaluation_benchmarking(dataset_name):
        model = model_args.model_name_or_path
        run_lm_evaluation_benchmarking(dataset_name, model)  # TODO (@Jipeng)
    else:
        raise NotImplementedError(
            f"benchmarking dataset {dataset_name} "
            " is not supported"
        )

if __name__ == "__main__":
    main()
