import json

autoif = json.load(open('/mnt/yizhenjia3/LMFlow/data/autoif/train.json', 'r'))

ifevallike = json.load(open('/mnt/yizhenjia3/LMFlow/data/ifeval-like/train.json', 'r'))

out = {"type": "conversation", "instances": autoif['instances'] + ifevallike['instances']}

print(len(autoif['instances']), len(ifevallike['instances']), len(out['instances']))

json.dump(out, open('/mnt/yizhenjia3/LMFlow/data/autoif_ifevallike/train.json', 'w'), indent=4)