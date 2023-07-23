import sentencepiece as spm
import sys
import os
dataset_path = sys.argv[1]+"/converted_data.txt"
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
mkdir("./output_models/new_tokenizer")
spm.SentencePieceTrainer.train('--input={} --model_prefix=./output_models/new_tokenizer/example --model_type=bpe --vocab_size=20000 --user_defined_symbols=0,1,2,3,4,5,6,7,8,9,%'.format(dataset_path))
sp = spm.SentencePieceProcessor()
sp.load('example.model')