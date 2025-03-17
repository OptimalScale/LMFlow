import json
import random
import openai
from typing import List
import sys
import os
from datasets import load_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from call_openai import get_openai_client
from tqdm import tqdm

client = get_openai_client()

question_set = "hybridial" # "topiocqa" # "inscit" # "coqa"

dataset_test = load_dataset("nvidia/ChatRAG-Bench", question_set, split="dev")

print(f"Number of test samples: {len(dataset_test)}\n")

chatrag_prompt = """
You are an expert at creating new question-answer pairs from given contexts.

Your task is to generate question-answer pairs based on a given context. Some exemplary question-answer pairs are provided for your reference. Your tasks are to:
(1) Augment each given question-answer pair with one or three wrong answers. If the original answer is yes/no, you only need to generate the opposite answer as the wrong answer. For other types of answers, you need to generate three wrong answers that are relevant to the context and question.
(2) Generate 5 more question-answer pairs. For each pair, include one correct answer and three wrong answers. If the question requires a yes/no response, include only one wrong answer. All questions and answers should be relevant to the given context and follow the style of the reference question-answer pairs.
(3) Format your output in JSON format with the following structure:
   {   
       "question": "...",
       "ability": "question-answering",
       "choices": ["correct answer", "wrong answer 1", "wrong answer 2", "wrong answer 3"],
       "answer": 0,
       "correct_answer_content": "correct answer",
       "id": "unique_identifier"
   }
"""

synthesized_data = []

for i in tqdm(range(len(dataset_test))):
    current_example = dataset_test[i]
    
    if i == len(dataset_test) - 1 or dataset_test[i]["ctxs"] != dataset_test[i + 1]["ctxs"]:
        samples = dict(current_example)
    else:
        continue
    
    context_text = samples["ctxs"][0]['text']
    reference_qa_pairs = []
    
    # for k in range(0, len(samples["messages"]), 2):
    #     reference_qa_pairs.append({
    #         "question": samples["messages"][k]['content'],
    #         "answer": samples["messages"][k+1]['content']
    #     })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": chatrag_prompt},
                {"role": "user", "content": json.dumps({
                    "context": context_text,
                    # "reference_qa": reference_qa_pairs
                })}
            ]
        )
        
        response_content = response.choices[0].message.content.strip()

        extracted_entries = []
    
        while "question" in response_content:
            entry_start = response_content.find('{')
            entry_end = response_content.find('}', entry_start) + 1
            
            if entry_start == -1 or entry_end == -1:
                break
            
            entry_str = response_content[entry_start:entry_end]
            response_content = response_content[entry_end:].strip()
            
            try:
                entry = json.loads(entry_str)
                entry = {"ctx": context_text, **entry}
                extracted_entries.append(entry)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error processing response: {e}")
        continue
    
    synthesized_data.extend(extracted_entries)
    

output_file = f"synthesized_chatrag_{question_set}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(synthesized_data, f, indent=4, ensure_ascii=False)

print(f"Saved synthesized data to {output_file}")
