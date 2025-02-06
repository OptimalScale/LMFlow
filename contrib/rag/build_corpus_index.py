import pickle
import os
from transformers import AutoTokenizer, AutoModel
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import torch
import faiss
import numpy as np
@dataclass
class RetrieverArguments:
    corpus_path: str = field(
        metadata={
            "help": "Please specify the path to the document corpus."
        }
    )

    embedding_type: Optional[str] = field(
        default="dpr",
        metadata={
            "help": "Please specify the type of retriever: bm25, or dpr"
        }
    )
    splitter: Optional[str] = field(
        default="\n\n",
        metadata={
            "help": "Please specify the splitter of your document."
        }
    )
    
    data_index_path: Optional[str] = field(
        default = './data/corpus',
        metadata={
            "help": "Please specify the name of data index name."
        }
    )

    device: int = field(
        default=0,
        metadata={
            "help": "The machine rank of gpu is used."
        }
    )

    

parser = HfArgumentParser((RetrieverArguments))
retriever_args = parser.parse_args_into_dataclasses()[0]
with open(retriever_args.corpus_path) as f:
    text = f.read()
texts = text.split(retriever_args.splitter)

if retriever_args.embedding_type == 'dpr':
    model_name = 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
    device = torch.device(f'cuda:{retriever_args.device}')
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    model = AutoModel.from_pretrained('sentence-transformers/facebook-dpr-question_encoder-single-nq-base').to(device)
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)
    def cls_pooling(model_output):
        return model_output[0][:,0]
    embeddings = cls_pooling(model_output)


    dim = 768
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.cpu().numpy())
    chunks = faiss.serialize_index(index)
    with open(retriever_args.data_index_path+'.dpr_index', "wb") as fp:
        pickle.dump(texts, fp)
        pickle.dump(chunks, fp)
    
elif retriever_args.embedding_type == 'bm25':
    with open(retriever_args.data_index_path+'.bm25_index', "wb") as fp:
        pickle.dump(texts, fp)
else:
    raise ValueError('The embedded method is not implemented. \
    Please specify the type of document embedding as one of the choices, [dpr, bm25].')






