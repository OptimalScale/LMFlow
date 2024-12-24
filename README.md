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
  - [Support](#support)
  - [License](#license)
  - [Citation](#citation)

## Quick Start

### Setup

Our package has been tested on Linux OS (Ubuntu 20.04). Other OS platforms (MacOS, Windows) are not fully tested, where you may encounter unexpected errors.

```bash
git clone -b v0.0.9 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```

> [!TIP]
> We use WandB to track and visualize the training process by default. Before running the training scripts, users may need to log in to WandB using the command: 
>```bash
>wandb login
>```
> For detailed instructions, refer to the [WandB Quickstart Guide](https://docs.wandb.ai/quickstart/). Step 1 (registration) and Step 2 (login using your WandB API key) should be sufficient to set up your environment.
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
bash run_finetune_with_lora.sh
```

> [!TIP]
> <details><summary>Merge LoRA Weight</summary>
>
>Merge LoRA weight and the base model into one using:  
>```sh
>bash ./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>


### Evaluation
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
@article{dong2023raft,
  title={Raft: Reward ranked finetuning for generative foundation model alignment},
  author={Dong, Hanze and Xiong, Wei and Goyal, Deepanshu and Pan, Rui and Diao, Shizhe and Zhang, Jipeng and Shum, Kashun and Zhang, Tong},
  journal={arXiv preprint arXiv:2304.06767},
  year={2023}
}
```
```
@article{pan2024lisa,
  title={LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning}, 
  author={Pan, Rui and Liu, Xiang and Diao, Shizhe and Pi, Renjie and Zhang, Jipeng and Han, Chi and Zhang, Tong},
  journal={arXiv preprint arXiv:2403.17919},
  year={2024}
}
```
