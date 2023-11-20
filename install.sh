#!/bin/bash

pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

gpu_state="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "${gpu_state}" == *"A100"* || "${gpu_state}" == *"A40"* || "${gpu_state}" == *"A6000"* ]]; then
  pip install flash-attn==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
