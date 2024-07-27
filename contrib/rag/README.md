# Retrieval-augmented generation

LMFlow now supports retrieval-augmented generation. We offer four different retrieval methods which include  DPR embeddings and BM25. Also, any model supported by LMFlow can be used for generation.

* DPR(Dense Passage Retrieval) Embeddings: \
https://arxiv.org/pdf/2004.04906
* BM25 retriever: \
https://python.langchain.com/v0.2/docs/integrations/retrievers/bm25/

## Requirements
Faiss library is required for dataset indexing.
```
pip install faiss-cpu pickle rank_bm25
```

## Build indexing for custom corpus for retrieval
If you want to use your own corpus for retrieval, first use `build_corpus_index.py` to build an index of the corpus embeddings. We offer one type of embedding method `dpr`and one retrieval method, `bm25`, which also requires indexing.

Below is an example that utilizes OpenAI embedding to index a corpus using '\n\n' as the splitter.

```
python ./scripts/build_corpus_index --corpus_path='corpus.txt' --splitter='\n\n' --embedding_type='dpr' --data_index_path='corpus'
```
Then it would save corpus and corpus index to ```corpus.dpr_index```.

## Inference and Evaluation

After building indexing of corpus, you can run the script `run_rag_inference.sh` that user can directly input question, and the script `run_rag_evaluation.sh` that user can input the path of dataset.

Here are two examples of each script.

```
bash ./scripts/run_rag_inference.sh --retriever_type='dpr' --corpus_index_path='corpus.dpr_index' --top_k_retrieve=5
```

```
bash ./scripts/run_rag_evaluation.sh --retriever_type='dpr' --corpus_index_path='corpus.dpr_index' --top_k_retrieve=5
```

## Known issue

Current `build_corpus_index.py` has memory issue, since it would load all corpus into memory at once, so if the size of corpus is larger than your memory, the process would be broken. Our next step is to enable our program to load corpus piece by piece, so that memory would not be an issue. Also, 


