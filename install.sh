#!/bin/bash

pip install -e .

gpu_state="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ *"A100"* == "${gpu_state}" -o *"A40"* == "${gpu_state}" ]]; then
  echo "YES!!!!!"
  pip install flash-attn==2.0.2
fi
