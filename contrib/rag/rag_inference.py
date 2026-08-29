#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings
import pickle
import faiss
import pickle
import torch
from rank_bm25 import BM25Okapi

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoModel as AM
from transformers import AutoTokenizer as AT
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments




logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class RAGInferArguments:
    retriever_type: Optional[str] = field(
        default='dpr',
        metadata={
            "help": "Please specify the type of document embedding: dpr, or bm25"
        }
    )
    corpus_index_path: Optional[str] = field(
        default="corpus.dpr_index",
        metadata={
            "help": "Please specify the path of corpus index. If you select wiki search, you do not need specify it."
        }
    )
    prompt_structure: Optional[str] = field(
        default="Answer the following question based on the background information.\n\nQuestion:{input_text}\n\nBackground:{background}\n\nAnswer:",
        metadata={
            "help": "prompt structure given user's input text and background information."
        },
    )
    top_k_retrieve: int = field(
        default=5,
        metadata={
            "help": "Please specify the number of the most relevant documents to be retrieved."
        }
    )

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
        RAGInferArguments
    ))
    model_args, pipeline_args, rag_args = parser.parse_args_into_dataclasses()
    inferencer_args = pipeline_args

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
        use_accelerator=True,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Inferences
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"
    
    top_k = rag_args.top_k_retrieve
    if rag_args.retriever_type == 'dpr':
        with open(rag_args.corpus_index_path, 'rb') as fp:
            corpus = pickle.load(fp)
            index = faiss.deserialize_index(pickle.load(fp))

        model_name = 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
        tokenizer = AT.from_pretrained(model_name)
        embed = AM.from_pretrained(model_name)
        def cls_pooling(model_output):
            return model_output[0][:,0]

    elif rag_args.retriever_type == 'bm25':
        with open(rag_args.corpus_index_path, "rb") as fp:
            corpus = pickle.load(fp)
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        raise ValueError('The type of retriever you specify is not implemented. Please specify it as one of [openai_embed, dpr_embed, wiki]')
    
    while True:
        input_text = input("User >>> ")
        prompt = rag_args.prompt_structure
        
        if rag_args.retriever_type == 'dpr':
            encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = embed(**encoded_input)
            embeddings = cls_pooling(model_output).numpy()

            _, ids = index.search(embeddings, k=top_k)
            docs = [corpus[int(id)] for id in ids[0]]
        elif rag_args.retriever_type == 'bm25':
            tokenized_query = input_text.split()
            docs = bm25.get_top_n(tokenized_query, corpus, n=top_k)

        background = '\n'.join(docs)
        background = background[-(model.get_max_length()-len(input_text)-len(prompt)):]
        all_input = prompt.format(input_text=input_text, background=background)
        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": all_input } ]
        })
        output_dataset = inferencer.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=inferencer_args.max_new_tokens,
            temperature=inferencer_args.temperature,
        )
        output = output_dataset.to_dict()["instances"][0]["text"]
        print('Bot:')
        print(output)


if __name__ == "__main__":
    main()
