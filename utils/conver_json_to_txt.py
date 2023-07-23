import json
import re
from pathlib import Path
import sys

dataset_path = sys.argv[1]
outputs = []
data_files = [
                x.absolute().as_posix()
                 for x in Path(dataset_path).glob("*.json")
            ]

for file_name in data_files:
    with open(file_name) as fin:
        json_data = json.load(fin)
        type = json_data["type"]
        for line in json_data["instances"]:
            outputs.append(line["text"])
            

outputfile = dataset_path + "/converted_data.txt"
with open(outputfile, "w") as f:
    for line in outputs:
        f.write(line)

print(f"convert dataset {dataset_path} to txt success.\n")

