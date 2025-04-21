import json

from datasets import load_dataset


data = load_dataset("argilla/ifeval-like-data", "default", split="train")

data_out = {"type": "conversation", "instances": []}
for datapoint in data:
    data_out["instances"].append(
        {
            "messages": [
                {"role": "user", "content": datapoint["instruction"]},
                {"role": "assistant", "content": datapoint["response"]},
            ]
        }
    )
    
len(data_out["instances"])
json.dump(
    data_out,
    open("data/ifeval-like-default/train.json", "w"),
    indent=4,
)