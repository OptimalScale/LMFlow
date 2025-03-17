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

# Define input and output file names
input_files = {
    "coin_flip": "coin_flip.json",
    "multiarith": "multiarith.json",
    "singleeq": "singleeq.json"
}

output_files = {
    "coin_flip": "coin_flip_new.json",
    "multiarith": "multiarith_new.json",
    "singleeq": "singleeq_new.json"
}

# Define the prompt template
prompt_template = """
You are an expert at creating new question-answer pairs based on similar examples.
Your task is to generate question-answer pairs following the provided question-answer pair example of a {task_name} task.

Your task is to: Generate 200 more question-answer pairs. For each question, generate one correct answer and three wrong answers. The wrong answers should have errors in either the thinking process or the final conclusion.

Example Entries:
{examples}

Please return the generated entries in JSON format with the same structure as the examples without any other special characters or symbols.
"""

# Process each file
for task_name, input_file in tqdm(input_files.items()):
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Select a few examples for context
    example_entries = json.dumps(data[:20], indent=4, ensure_ascii=False)
    
    try:
        # Generate new entries
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_template.format(task_name=task_name, examples=example_entries)}
            ]
        )
        
        response_content = response.choices[0].message.content.strip()
        
        generated_entries = []
    
        while "question" in response_content or "sQuestion" in response_content:
            entry_start = response_content.find('{')
            entry_end = response_content.find('}', entry_start) + 1
            
            if entry_start == -1 or entry_end == -1:
                break
            
            entry_str = response_content[entry_start:entry_end]
            response_content = response_content[entry_end:].strip()
            
            try:
                entry = json.loads(entry_str)
                generated_entries.append(entry)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error processing response for {task_name}: {e}")
    
    # Save new entries to output file
    output_file = output_files[task_name]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_entries, f, indent=4, ensure_ascii=False)
    
    print(f"Saved generated data to {output_file}")