# Position Interpolation 
Now LMFlow supports the latest Linear & NTK (Neural Kernel theory) scaling techniques for LLaMA models. \
For more details of these techniques, you can checkout the links below:
* Linear scaling: \
https://arxiv.org/abs/2306.15595
* NTK scaling: \
https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
## Usage
To use the Position Interpolation Techniques, you need to set the following options:
```
--truncate_to_model_max_length False
--do_rope_scaling True
```
For linear scaling, set the extending ratio by:
```
--rope_pi_ratio 4
```
For NTK scaling, set the extending ratio by:
```
--rope_ntk_ratio 4
```
Here is an example of evaluation bash code:
```
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path pinkmanlove/llama-7b-hf \
    --dataset_path data/wiki_en_eval \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --truncate_to_model_max_length False \
    --block_size 4096 \
    --use_flash_attention True \
    --do_rope_scaling True \
    --rope_pi_ratio 2 \
    --rope_ntk_ratio 4 \
    --metric ppl
```