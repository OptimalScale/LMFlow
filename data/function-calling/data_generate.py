import json
import random
import openai
from typing import List

def generate_function_distractors(correct_answer: str, function_desc: str, question: str) -> List[str]:
    """Generate plausible but incorrect function calls as distractors"""
    try:
        messages = [
            {
                "role": "system", 
                "content": """You are an expert at creating plausible but incorrect function calls.

Your task is to generate incorrect function calls that look reasonable but contain common mistakes, such as:
- Using wrong parameter names (e.g., using 'length' instead of 'width')
- Missing required parameters
- Adding unnecessary parameters
- Using wrong data types for parameters (e.g., string instead of number)
- Mixing up similar function names (e.g., 'calc_area' vs 'calculate_area')
- Using incorrect values for parameters

Generate exactly 3 different incorrect function calls that a programmer might mistakenly write.
Each should be a complete function call with parameters.
Make the mistakes subtle and believable - they should look plausible at first glance.
Separate responses with |||

Example:
If the correct call is: calculate_area(width=10, height=5)
You might generate:
calculate_area(length=10, breadth=5)  ||| 
area_calculate(width=10, height=5)  |||
calculate_area(width="10", height=5, scale=1)"""
            },
            {
                "role": "user", 
                "content": f"""Function Description: {function_desc}
Question to solve: {question}
Correct function call: {correct_answer}

Generate 3 plausible but incorrect function calls that look similar to the correct answer.
Each should be a complete function call that a programmer might mistakenly write.
Separate each response with |||."""
            }
        ]

        # 打印传给OpenAI的消息
        print("\n=== Messages sent to OpenAI ===")
        for msg in messages:
            print(f"\n{msg['role'].upper()}:")
            print(msg['content'])
        print("=" * 50)

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8
        )
        
        distractors = response.choices[0].message.content.split('|||')
        distractors = [d.strip() for d in distractors[:3]]
        
        # 确保有3个干扰项
        if len(distractors) < 3:
            distractors.extend([
                f"similar_function({correct_answer.split('(')[1]}",
                f"{correct_answer.split('(')[0]}(wrong_param=value)",
                f"{correct_answer.split('(')[0]}()"
            ][:3 - len(distractors)])
        
        return distractors[:3]
    except Exception as e:
        print(f"Error generating distractors: {e}")
        return [
            f"similar_function({correct_answer.split('(')[1]}",
            f"{correct_answer.split('(')[0]}(wrong_param=value)",
            f"{correct_answer.split('(')[0]}()"
        ]

def process_json_file(file_path, answer_file_path):
    # 存储处理后的结果
    processed_data = []
    
    # 读取答案文件
    answers = {}
    with open(answer_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ground_truth = data['ground_truth'][0]
            function_name = list(ground_truth.keys())[0]
            params = ground_truth[function_name]
            # 构建答案字符串
            answer = f"{function_name}("
            param_strs = []
            for param, values in params.items():
                value = values[0] if isinstance(values, list) else values
                param_strs.append(f"{param}={value}")
            answer += ", ".join(param_strs) + ")"
            answers[data['id']] = answer
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                question = data['question'][0]['content']
                function = data['function'][0]
                function_str = (f"{function['name']} with description: {function['description']} "
                              f"and parameters: {function['parameters']}")
                
                # 获取正确答案
                correct_answer = answers[data['id']]
                
                # 生成干扰项
                distractors = generate_function_distractors(
                    correct_answer=correct_answer,
                    function_desc=function_str,
                    question=question
                )
                
                # 随机选择正确答案的位置
                options = ['A', 'B', 'C', 'D']
                correct_position = random.choice(options)
                
                output = f"Given the following function: {function_str}, if you are asked to {question}, you will call...\n"
                
                # 添加选项
                current_distractor = 0
                for option in options:
                    if option == correct_position:
                        output += f"{option}. {correct_answer}\n"
                    else:
                        output += f"{option}. {distractors[current_distractor]}\n"
                        current_distractor += 1
                
                output += f"\nCorrect Answer: {correct_position}"
                
                processed_data.append(output)
                
            except Exception as e:
                print(f"Error processing line: {e}")
    
    return processed_data

# 使用示例
file_path = "/lustre/fsw/portfolios/llmservice/users/sdiao/data-challenge/LMFlow/data/function-calling-raw/BFCL_v2_simple.json"
answer_file_path = "/lustre/fsw/portfolios/llmservice/users/sdiao/data-challenge/LMFlow/data/function-calling-raw/possible_answer/BFCL_v2_simple.json"
results = process_json_file(file_path, answer_file_path)

# 打印前几个结果作为示例
for i, result in enumerate(results[:3]):
    print(f"\nExample {i+1}:")
    print(result)
    print("-" * 80)

# 保存到文件
with open('processed_output.txt', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(result + "\n\n" + "-"*80 + "\n\n")