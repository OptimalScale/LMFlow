import json
import openai
from typing import List
import random

def generate_distractors(correct_answer: str, context: str) -> List[str]:
    """Generate responses from dramatically different character backgrounds"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
Separate responses with |||"""
                },
                {
                    "role": "user", 
                    "content": f"""Original Role Description and Context: {context}
Response from the specified role: {correct_answer}

Generate 3 responses from completely different character backgrounds.
Each response should address the same question but from a drastically different perspective.
Separate each response with |||."""
                }
            ],
            temperature=0.9
        )
        
        distractors = response.choices[0].message['content'].split('|||')
        distractors = [d.strip() for d in distractors[:3]]
        
        if len(distractors) < 3:
            distractors.extend([
                "Speaking as someone from a completely different walk of life...",
                "From my vastly different background and experience...",
                "As someone with an entirely different perspective..."
            ][:3 - len(distractors)])
        
        return distractors[:3]

def process_roleplay_data(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 修改数据处理部分
    processed_data = []
    
    for instance in data['instances']:
        system_prompt = instance['system']
        user_message = instance['messages'][0]['content']
        correct_answer = instance['messages'][1]['content']
        
        combined_input = f"{system_prompt}\n{user_message}"
        
        # 生成干扰选项
        distractors = generate_distractors(correct_answer, combined_input)
        
        # 创建选项列表并随机打乱
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 找出正确答案的索引
        correct_index = options.index(correct_answer)
        
        # 将处理后的数据添加到列表中
        processed_data.append({
            'question': combined_input,
            'options': options,
            'correct_index': correct_index,
            'correct_answer': correct_answer
        })
    
    return processed_data
    

def save_processed_data(processed_data, output_file_path):
    # 将处理后的数据保存为JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'data': processed_data
        }, f, ensure_ascii=False, indent=2)


openai.api_key = 'your-api-key-here'  # 请替换为你的API密钥


# 使用示例
input_file_path = '/lustre/fsw/portfolios/llmservice/users/sdiao/data-challenge/LMFlow/data/roleplay-raw/rolebench-test-role_generalization_english.json'
output_file_path = '/lustre/fsw/portfolios/llmservice/users/sdiao/data-challenge/LMFlow/data/roleplay/roleplay_test.json'

# 处理数据
processed_data = process_roleplay_data(input_file_path)

# 保存处理后的数据
save_processed_data(processed_data, output_file_path)

# 打印确认信息
print(f"Processed {len(processed_data)} instances")
print(f"Data saved to: {output_file_path}")

# 修改打印示例的部分
for i, example in enumerate(processed_data[:2]):
    print(f"\nExample {i+1}:")
    print("Question:", example['question'][:200] + "...")
    print("Options:")
    for j, option in enumerate(example['options']):
        print(f"{chr(65+j)}. {option[:100]}...")
    print(f"Correct Answer: {chr(65+example['correct_index'])}")