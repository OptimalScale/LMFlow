# Speculative Decoding
## Introduction
[Speculative Decoding (Ref: arXiv:2211.17192v2)](https://arxiv.org/abs/2211.17192) is now available for playing via:
```bash
python ./examples/speculative_inference.py \ 
  --model            # your_model_name_or_path 
  --draft_model      # your_draft_model_name_or_path 
  --temperature      # your_temperature 
  --gamma            # your_gamma
  --max_new_tokens   # your_max_new_tokens
  --gpu              # your_gpu_id
```
For example, 
```bash
python ./examples/speculative_inference.py \ 
  --model gpt2-xl 
  --draft_model gpt2 
  --temperature 0.3 
  --gamma 5
  --max_new_tokens 512
  --gpu 0
```
Another example,
```bash
python ./examples/speculative_inference.py \ 
  --model /home/eric/Documents/models/gpt2-xl 
  --draft_model /home/eric/Documents/models/gpt2 
  --temperature 0 
  --gamma 3
  --max_new_tokens 1024
  --gpu 7
```
## Parameter Instruction
`model`, `draft_model`
- Huggingface model name or locally cached model path.
-  Currently only supports huggingface decoder only models. 
-  `model` refers to the target model (i.e., the large model you want to accelerate) in the paper. 
-  `draft_model` refers to the draft model in the paper.

`temperature`
- Temperature for sampling. When temperature <= 1e-6, will use argmax sampling.

`gamma`
- Number of tokens that the draft model will generate at each step. See the paper for more details.

`max_new_tokens`
- Maximum number of tokens that the speculative inference will generate.
- TODO: currently the speculative decoding will always generate `max_new_tokens` tokens. We will add a `stop_token` in the future.

`gpu`
- gpu id, currently speculative inference only support single gpu.

## Experiments
We tested the speculative inference using the first 100 inputs from alpaca test dataset as prompts. When `model=gpt2-xl`, `draft_model=gpt2`, `temperature=0.`, `max_new_tokens=512`, we observed the following acceleration:

|gamma|speedup (inference time)|speed up (num of forwards)
|--|--|--|
|1|1.75x|1.96x|
|2|2.29x|2.89x|
|3|2.71x|3.77x|
|4|3.06x|4.63x|
|5|3.35x|5.44x|
|6|3.65x|6.23x|
|7|3.82x|6.94x|
|8|3.96x|7.64x|
|9|4.05x|8.33x|
|10|4.14x|9.00x|

Note that the speedup may be overestimated. When `temperature=0`, `gpt2-xl` and `gpt2` tend to generate duplicated tokens as the number of tokens generated increases, thus making the target model more likely to accept the draft model's output.