"""The program includes several functions: setting a random seed, 
loading data from a JSON file, batching data, and extracting answers from generated text.
"""

import json
import os
import random
import re
from typing import Union, List, TypedDict, Dict

import numpy as np
import torch


def set_random_seed(seed: int):
    """
    Set the random seed for `random`, `numpy`, `torch`, `torch.cuda`.

    Parameters
    ------------
    seed : int
        The default seed.
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(file_name: str):
    """
    Load data with file name.

    Parameters
    ------------
    file_name : str.
        The dataset file name.
    
    Returns
    ------------
    inputs : list.
        The input texts of the dataset.
    outputs : list.
        The output texts file datasets.    
    len : int.
        The length of the dataset.
    """
    inputs = []
    outputs = []
    type = ""
    with open(file_name, encoding='utf-8') as f:
        json_data = json.load(f)
        type = json_data["type"]
        for line in json_data["instances"]:
            inputs.append(line["input"])
            outputs.append(line["output"])

    print(f"load dataset {file_name} success.\n")
    print(f"Type : {type}, datasize : {len(outputs)}")

    return inputs, outputs, len(outputs)

def batchlize(examples: list, batch_size: int, random_shuffle: bool):
    """
    Convert examples to a dataloader.

    Parameters
    ------------
    examples : list.
        Data list.
    batch_size : int.

    random_shuffle : bool
        If true, the dataloader shuffle the training data.
    
    Returns
    ------------
    dataloader:
        Dataloader with batch generator.
    """
    size = 0
    dataloader = []
    length = len(examples)
    if (random_shuffle):
        random.shuffle(examples)
    while size < length:
        if length - size > batch_size:
            dataloader.append(examples[size : size+batch_size])
            size += batch_size
        else:
            dataloader.append(examples[size : size+(length-size)])
            size += (length - size)
    return dataloader


def preview_file(file_path: str, chars: int = 100):
    """
    Returns the first and last specified number of characters from a file
    without loading the entire file into memory, working with any file type.
    
    Args:
        file_path (str): Path to the file to be previewed
        chars (int, optional): Number of characters to show from start and end. Defaults to 100.
    
    Returns:
        tuple: (first_chars, last_chars) - The first and last characters from the file
    """
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        first_chars = f.read(chars)
        
        if file_size <= 2 * chars:
            return first_chars, ""
        
        last_chunk_position = max(0, file_size - chars)
        
        f.seek(0)
        f.seek(last_chunk_position)
        
        last_chars = f.read(chars)
        
    return first_chars, last_chars


def get_dataset_type_fast(file_path: str, max_chars: int = 100) -> Union[str, None]:
    '''Get the type values from the first and last n lines of a large json dataset.
    '''
    file_content_preview = []
    dataset_type = None
    dataset_type_pattern = re.compile(r'[\"\']type[\"\']:\s*[\'\"]([^"]+)[\'\"]')
    file_content_preview.extend(preview_file(file_path, max_chars))
    for content in file_content_preview:
        try:
            dataset_type = dataset_type_pattern.search(content).group(1)
            break
        except AttributeError:
            continue
    return dataset_type


def check_dataset_instances_key_fast(file_path: str, instances_key: str, max_lines: int = 100) -> bool:
    '''Check if the dataset instances key matches the instance_key.
    '''
    file_content_preview = []
    instance_key_pattern = re.compile(r'[\"\']' + instances_key + r'[\"\']')
    file_content_preview.extend(preview_file(file_path, max_lines))
    for content in file_content_preview:
        if instance_key_pattern.search(content):
            return True
    return False


def answer_extraction(response, answer_type=None):   #use this funtion to extract answers from generated text

    """
    Use this funtion to extract answers from generated text

    Parameters
    ------------
    args : 
        Arguments.
    response : str
        plain string response.


    Returns
    ------------
    answer:
        Decoded answer (such as A, B, C, D, E for mutiple-choice QA).
    """

    # temp = response["generated_text"]
    temp = response
    if answer_type in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith", "math"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif answer_type in ("aqua", "csqa", "multiple_choice"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif answer_type in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif answer_type in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif answer_type in ("pubmedqa", "binary_choice"):
        # pattern = "Output: (yes|no|maybe)"
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:] if sttr is not None else "N/A"
        pattern = "(answer|Answer|ANSWER|output|Output|OUTPUT|A): \(*(yes|Yes|YES|no|No|NO|maybe|Maybe|MAYBE)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            mid_answer = mid_answer.split(":")[-1].strip()
            answer = mid_answer.lower()
        else:
            pattern = "(yes|Yes|YES|no|No|NO|maybe|Maybe|MAYBE)(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                answer = sttr.group(0)[:-1].lower()
            else:
                answer = "N/A"
        return answer
    elif answer_type == "medmcqa":
        # pattern = "Output: (A|B|C|D)."
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:-1].lower() if sttr is not None else "N/A"
        pattern = "(answer|Answer|ANSWER|output|Output|OUTPUT|A): \(*(A|B|C|D|a|b|c|d)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            answer = mid_answer[-1].lower()
        else:
            pattern = "\(*(A|B|C|D|a|b|c|d)\)*(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                if '(' in sttr.group(0):
                    answer = sttr.group(0)[1].lower()
                else:
                    answer = sttr.group(0)[0].lower()
            else:
                answer = "N/A"
        return answer

    elif answer_type == "usmle":
        # pattern = "Output: (A|B|C|D)."
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:-1].lower() if sttr is not None else "N/A"
        pattern = "(Answer|Output|A): \(*(A|B|C|D|a|b|c|d)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            answer = mid_answer[-1].lower()
        else:
            pattern = "\(*(A|B|C|D|a|b|c|d)\)*(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                if '(' in sttr.group(0):
                    answer = sttr.group(0)[1].lower()
                else:
                    answer = sttr.group(0)[0].lower()
            else:
                answer = "N/A"
        return answer
    elif answer_type == "text":
        return response
    else:
        raise NotImplementedError(f"Unsupported answer type: {answer_type}")

    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

            # round the answer to nearest integer
        if answer_type in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif answer_type in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
    else:
        answer = ""
    return answer


def process_image_flag(text, image_flag="<ImageHere>"):
    texts = text.split(image_flag)
    if len(texts) > 1:
        image_token_indexes = [len(text) for text in texts[:-1]]
    else:
        image_token_indexes = []
    # cumsun
    image_token_indexes = list(np.cumsum(image_token_indexes))
    texts = "".join(texts)
    return texts, image_token_indexes


class VLLMInferenceResultWithInput(TypedDict):
    input: str
    output: Union[List[str], List[List[int]]]
    

class RewardModelInferenceResultWithInput(TypedDict):
    input: str
    output: List[Dict[str, Union[str, float]]] # [{"score": 0.5, "text": "output text"}]