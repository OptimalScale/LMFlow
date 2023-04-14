import transformers

from transformers import AutoTokenizer
from datasets import Dataset

import json
import os
from tqdm import tqdm

def read_jsonl(data_path):
    with open(data_path,'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(data_path, data):
    with open(data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
def read_json(data_path):
    with open(data_path,'r') as f:
        data = json.load(f)
    return data

def write_json(data_path, data):
    with open(data_path,'w',encoding='utf8') as f:
        json.dump(data,f)

hardcode_json_path = "/home/home/jzhanggr/LMFlow/data/hardcode/train/hardcode_v1.json"


my_list = read_json(hardcode_json_path)['instances']
raw_datasets = Dataset.from_list(my_list)

tokenizer = AutoTokenizer.from_pretrained("/home/home/Tribbiani/vicuna-7b")

text_column_name = "text"
def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=40,
    # remove_columns=column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

# for i, item in tqdm(enumerate(data)):
#     #print(data)
#     text = item['text']
#     inputs = tokenizer.encode(text, return_tensors="pt").to(device=0)
#     # print(str(i) + " finished")




# tokenizer.encode('Input: ' + question + ' \n Output:', return_tensors="pt").to(device=local_rank)