import json
import os

from datasets import load_dataset

if_mix_ds = load_dataset("wheresmyhair/instruction-following-mix-deepseek-34k", split='train')
if_mix_raw_34k = load_dataset("wheresmyhair/instruction-following-mix-raw-34k", split='train')
if_mix_raw_340k = load_dataset("wheresmyhair/instruction-following-mix-raw-340k", split='train')
autoif = load_dataset("wheresmyhair/ultrachat-autoif", split='train')

out = {"type": "conversation", "instances": []}
for data in autoif:
    out['instances'].append(data)
os.mkdir("./data/autoif")
json.dump(out, open("./data/autoif/train.json", "w"), indent=4)

out = {"type": "conversation", "instances": []}
for data in if_mix_ds:
    out['instances'].append(data)
os.mkdir("./data/if_mix_ds")
json.dump(out, open("./data/if_mix_ds/train.json", "w"), indent=4)

out = {"type": "conversation", "instances": []}
for data in if_mix_raw_34k:
    out['instances'].append(data)
os.mkdir("./data/if_mix_raw_34k")
json.dump(out, open("./data/if_mix_raw_34k/train.json", "w"), indent=4)

out = {"type": "conversation", "instances": []}
for data in if_mix_raw_340k:
    out['instances'].append(data)
os.mkdir("./data/if_mix_raw_340k")
json.dump(out, open("./data/if_mix_raw_340k/train.json", "w"), indent=4)

out = {"type": "conversation", "instances": []}
for data in if_mix_raw_34k:
    out['instances'].append(data)
for data in autoif:
    out['instances'].append(data)
os.mkdir("./data/if_mix_raw_34k_autoif")
json.dump(out, open("./data/if_mix_raw_34k_autoif/train.json", "w"), indent=4)