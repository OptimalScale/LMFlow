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
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser
from transformers import AutoModel as AM
from transformers import AutoTokenizer as AT
from dataclasses import dataclass, field
from typing import Optional
import torch

import faiss
import pickle
from rank_bm25 import BM25Okapi


from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

@dataclass
class RAGEvalArguments:
    retriever_type: Optional[str] = field(
        default='dpr',
        metadata={
            "help": "Please specify the type of document embedding: dpr, and bm25"
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
            "help": "prompt structure given user's input text."
        },
    )
    top_k_retrieve: int = field(
        default=5,
        metadata={
            "help": "Please specify the number of the most relevant documents to be retrieved."
        }
    )

pipeline_name = "evaluator"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments, RAGEvalArguments))
model_args, data_args, pipeline_args, rag_args = parser.parse_args_into_dataclasses()

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(
    model_args, 
    tune_strategy='none', 
    ds_config=ds_config, 
    use_accelerator=pipeline_args.use_accelerator_for_evaluator
)
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

dataset = Dataset(data_args)

data_dict = dataset.to_dict()

for i, instance in enumerate(data_dict["instances"]):
    input_text = instance['input']
    
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
    all_input = rag_args.prompt_structure.format(input_text=input_text, background=background)
    data_dict["instances"][i] = all_input

dataset = dataset.from_dict(data_dict)

evaluator = AutoPipeline.get_pipeline(
    pipeline_name=pipeline_name,
    model_args=model_args,
    data_args=data_args,
    pipeline_args=pipeline_args,
)
evaluator.evaluate(model=model, dataset=dataset, metric=pipeline_args.metric)
