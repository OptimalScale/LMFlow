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

For sanity check, we provide [a small dataset](./data/wikitext-2-raw-v1/) for you to test the finetuning process.

To process your own dataset, please refer to our [doc](https://optimalscale.github.io/LMFlow/examples/DATASETS.html).

### Training
[DoRA](https://arxiv.org/pdf/2402.09353) is a parameter-efficient finetuning algorithm and is more efficient than full finetuning.
```sh
bash train.sh
```
Note: Please double-check that you have updated the [training script](https://github.com/OptimalScale/LMFlow/blob/data4elm/train.sh) with the correct arguments for your use case.

Note: So that we eliminate hyperparameters as a confounding factor, you must keep `num_train_epochs` as `1`, `learning_rate` as `1e-5`, and `lora_r` as 16. 

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
    --model_args pretrained=[YOUR_MODEL_PATH],trust_remote_code=True,cache_dir=~/.cache \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/test_elmb
```

Note that in order to test your model, you must first upload it to HuggingFace. The [YOUR_MODEL_PATH] is the HuggingFace model path (`username/model_name`) to your uploaded model.

Thus, after finetuning your model and merging the DoRA weights, you must upload the model to HuggingFace.
You may reference `example_upload_peft_model.py` for a starter script on how to upload your DoRA-finetuned model.


## FAQ
Below are some commonly asked questions from our [Discord](https://discord.com/invite/TVjjdcbuFG).

**Q:** Where can I go if I have questions about the challenge?
**A:** The main place to ask questions will be under the `challenge-questions` text channel in our [Discord](https://discord.com/invite/TVjjdcbuFG).

**Q:** How can I resume from a checkpoint?
**A:** Users can resume from checkpoints by adding the argument  `--resume_from_checkpoint` to the training script with the path to the latest checkpoint.
For example, `--resume_from_checkpoint [model-dir]/checkpoint-[checkpoint-index]`.

**Q:** How can I test using a validation split?
**A:** Users can view validation loss during training by adding the arguments `--validation_split_percentage`, `--eval_strategy`, and `--eval_steps`. For instance:
`--validation_split_percentage 5 \ --eval_strategy steps \ --eval_steps 20` will show the validation loss every 20 steps using a validation split of 5 percent.

**Q:** How do I know if I am registered?
**A:** You will receive a confirmation email titled "PLEASE READ: Data Filtering Challenge - Confirmation of Registration" from an Outlook account named "data4elm". The names of the registered teams will also be listed on our Discord periodically.

**Q:** Where can I find the dataset?
**A:** You can find the starter dataset [here](https://huggingface.co/datasets/nvidia/ClimbLab).

**Q:** The starter dataset consists of tokens. How can I convert it into an LMFlow-friendly format?
**A:** You can use the [unofficial detokenized dataset](http://huggingface.co/datasets/OptimalScale/ClimbLab), or you may detokenize the dataset yourself using the script `detokenize_climblab.py` found [here](http://huggingface.co/datasets/OptimalScale/ClimbLabhuggingface.co/datasets/nvidia/ClimbLab).

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
