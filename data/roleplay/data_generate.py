import json
import openai
from typing import List
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from call_openai import get_openai_client
from tqdm import tqdm

def generate_distractors(correct_answer: str, context: str, client) -> List[str]:
    """Generate responses from dramatically different character backgrounds"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert at creating responses from contrasting character backgrounds.

    Your task is to generate responses from characters with COMPLETELY DIFFERENT backgrounds than the original role.
    For example, if the original role is a "professional doctor", you might respond as:
    - A teenage social media influencer
    - A traditional farmer from a rural village
    - A sci-fi spacecraft engineer

    Choose characters that are maximally different in terms of:
    - Professional background
    - Age group and life experience
    - Cultural and social context
    - Education level and expertise area

    Generate exactly 3 responses, each from a distinctly different character background.
    Make sure each character would naturally have a very different perspective on the topic.
    IMPORTANT: Do NOT include character labels like "Response from a [character type]:" - just write the response directly as that character would.
    Separate responses with |||"""
                },
                {
                    "role": "user", 
                    "content": f"""Original Role Description and Context: {context}
    Response from the specified role: {correct_answer}

    Generate 3 responses from completely different character backgrounds.
    Each response should address the same question but from a drastically different perspective.
    Do NOT include any character labels or introductions - just write the direct responses as if from those characters.
    Separate each response with |||."""
                }
            ],
            temperature=0.9
        )
        
        distractors = response.choices[0].message.content.split('|||')
        distractors = [d.strip() for d in distractors[:3]]
        
        if len(distractors) < 3:
            distractors.extend([
                "Speaking as someone from a completely different walk of life...",
                "From my vastly different background and experience...",
                "As someone with an entirely different perspective..."
            ][:3 - len(distractors)])
        
        return distractors[:3]
    except Exception as e:
        print(f"Error generating distractors: {e}")
        return []

def process_roleplay_data(json_file_path, client, NUM_EXAMPLES, output_file_path, save_interval=300):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 修改数据处理部分
    processed_data = []
    count = 0
    total = min(NUM_EXAMPLES, len(data['instances']))
    
    for instance in tqdm(data['instances'], total=total, desc="Processing instances"):
        if count >= NUM_EXAMPLES:
            break
        count += 1
        system_prompt = instance['system']
        user_message = instance['messages'][0]['content']
        correct_answer = instance['messages'][1]['content']
        
        combined_input = f"{system_prompt}\n{user_message}"
        
        # 生成干扰选项
        distractors = generate_distractors(correct_answer, combined_input, client)
        if len(distractors) < 3:
            client = get_openai_client()
            continue
        # 创建选项列表并随机打乱
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 找出正确答案的索引
        correct_index = options.index(correct_answer)
        
        # 将处理后的数据添加到列表中
        processed_data.append({
            'question': combined_input,
            'ability': 'roleplay',
            'choices': options,
            'answer': correct_index,
            'correct_answer_content': correct_answer,
            'id': count
        })
        
        # 每隔save_interval步保存一次数据
        if count % save_interval == 0:
            save_checkpoint(processed_data, output_file_path, count)
            print(f"Checkpoint saved: {count} instances processed")
    
    return processed_data

def save_checkpoint(data, output_file_path, count):
    """保存处理数据的检查点"""
    checkpoint_path = f"{output_file_path}.checkpoint_{count}"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

NUM_EXAMPLES = 10000
input_file_path = '/lustre/fsw/portfolios/llmservice/users/sdiao/data-challenge/LMFlow/data/roleplay-raw/rolebench-test-role_generalization_english.json'
output_file_path = '/lustre/fsw/portfolios/llmservice/users/sdiao/data-challenge/LMFlow/data/roleplay/roleplay_valid.json'

client = get_openai_client()

processed_data = process_roleplay_data(input_file_path, client, NUM_EXAMPLES, output_file_path, save_interval=300)

with open(output_file_path, 'w', encoding='utf-8') as f:
    for item in processed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Processed {len(processed_data)} instances")
print(f"Data saved to: {output_file_path}")

# # 修改打印示例的部分
# for i, example in enumerate(processed_data[:2]):
#     print(f"\nExample {i+1}:")
#     print("Question:", example['question'][:200] + "...")
#     print("Options:")
#     for j, option in enumerate(example['choices']):
#         print(f"{chr(65+j)}. {option[:100]}...")
#     print(f"Correct Answer: {chr(65+example['answer'])}")