# LMFlow

An extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community.

## Table of Contents
- [LMFlow](#lmflow)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Setup](#setup)
    - [Prepare Dataset](#prepare-dataset)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [FAQ](#faq)
  - [Support](#support)
  - [License](#license)
  - [Citation](#citation)

## Quick Start

### Setup

Our package has been tested on Linux OS (Ubuntu 20.04). Other OS platforms (MacOS, Windows) are not fully tested, where you may encounter unexpected errors.

```bash
git clone -b v0.0.9 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
git checkout data4elm
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```

> [!TIP]
> We use Wandb to track and visualize the training process by default. Before running the training scripts, users may need to log in to Wandb using the command: 
>```bash
>wandb login
>```
> For detailed instructions, refer to the [Wandb Quickstart Guide](https://docs.wandb.ai/quickstart/). Step 1 (registration) and Step 2 (login using your Wandb API key) should be sufficient to set up your environment.
>
> <details><summary>Disabling wandb</summary>  
>
> One can disable wandb by either:  
>
> 1. Adding environment variable before running the training command.
>
>```bash
>export WANDB_MODE=disabled
>```
>
> 2. OR, specifying the integrations to report the results and logs to. In the training script, add:
>
>```bash
>--report_to none \
>```
>
> </details>

### Prepare Dataset

For sanity check, we provide [a small dataset](./data/wikitext-2-raw-v1/test) for you to test the finetuning process.

To process your own dataset, please refer to our [doc](https://optimalscale.github.io/LMFlow/examples/DATASETS.html).

### Training
LoRA is a parameter-efficient finetuning algorithm and is more efficient than full finetuning.
```sh
bash train.sh
```
Note: Please double-check that you have updated the [training script](https://github.com/OptimalScale/LMFlow/blob/data4elm/train.sh) with the correct arguments for your use case.

> [!TIP]
> <details><summary>Merge Dora Weight</summary>
>
>Merge Dora weight and the base model into one using:  
>```sh
>bash ./scripts/run_merge_dora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/dora \
>  --output_model_path output_models/dora_merged \
>```
></details>


### Evaluation
Aligned with the objective of this challenge, we propose a new benchmark for evaluating edge LMs, named the Edge Language Model Benchmark (ELMB). 

It includes the following tasks: 

- Roleplay: Enhancing performance in interactive digital environments. 

- Reasoning: Improving complex problem-solving for downstream applications like robotics. 

- Function Calling: Optimizing models for mobile device interactions. 

- Retrieval-Augmented Generation (RAG): Boosting capabilities in retrieval-augmented applications. 

To evaluate the performance of the model on the ELMB, you can use the following command:
```bash
cd LMFlow/lm-evaluation-harness
pip install -e . 

lm_eval --model hf \
    --model_args pretrained=[YOUR_MODEL_PATH]],trust_remote_code=True,cache_dir=~/.cache \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/test_elmb
```

## FAQ
[TODO]

## Support

If you need any help, please submit a Github issue.

## License
The code included in this project is licensed under the [Apache 2.0 license](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE).
If you wish to use the codes and models included in this project for commercial purposes, please sign this [document](https://docs.google.com/forms/d/e/1FAIpQLSfJYcci6cbgpIvx_Fh1xDL6pNkzsjGDH1QIcm4cYk88K2tqkw/viewform?usp=pp_url) to obtain authorization.

## Citation
If you find this repository useful, please consider giving ⭐ and citing our [paper](https://arxiv.org/abs/2306.12420):

```
@article{diao2023lmflow,
  title={Lmflow: An extensible toolkit for finetuning and inference of large foundation models},
  author={Diao, Shizhe and Pan, Rui and Dong, Hanze and Shum, Ka Shun and Zhang, Jipeng and Xiong, Wei and Zhang, Tong},
  journal={arXiv preprint arXiv:2306.12420},
  year={2023}
}
```

```
@inproceedings{liu2024dora,
  title={Dora: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```