<p align="center" width="100%">
<img src="assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-orange.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/srGxyazbNs)
[![WeChat badge](https://img.shields.io/badge/微信-加入-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/02/iHtF23.png)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)


An extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community.

Large Language Model for All. See our [vision](https://github.com/OptimalScale/LMFlow/tree/shizhe-update-readme4#vision).
共建大模型社区，让每个人都能训得起大模型。查看我们的[愿景](https://github.com/OptimalScale/LMFlow/tree/shizhe-update-readme4#vision)。

<p align="center" width="100%">
<img src="assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## Latest News
* [2023-04-02] [Web service is online!](https://lmflow.com/)
* [2023-04-01] [Release Chinese checkpoints in model zoo: Hu (湖羊), Dongshan (东山羊), and Hetian (和田羊).](https://github.com/OptimalScale/LMFlow/tree/shizhe-update-readme4#model-zoo)
* [2023-04-01] [Release English checkpoints in model zoo: LLaMA7B-medical, LLaMA13B-medical, and LLaMA33B-medical.](https://github.com/OptimalScale/LMFlow/tree/shizhe-update-readme4#model-zoo)
* [2023-03-27] [Support full tuning and lora tuning for all decoder models.](https://github.com/OptimalScale/LMFlow/tree/shizhe-update-readme4#supported-models) 
* [2023-03-27] [Tasked tuned model beats ChatGPT on medical domain](https://github.com/OptimalScale/LMFlow#model-performance)
* [2023-03-27] [Release code and checkpoints - version 0.0.1](https://optimalscale.github.io/LMFlow/)

## Model Performance

|                |  PubMedQA (ID) | MedQA-USMLE (OOD) | MedMCQA (ID) |  Average |
|:---------:|:--------:|:-----------:|:-------:|:----:|
| Human (pass)   |  60.0   |     50.0    |         |      |
| Human (expert) |    78.0   |     87.0    |  90.0   | 85.0 |
|   |      |              |    |  |
|  InstructGPT 175B   |   73.2   |     46.0    |  44.0   | 54.4 |
|    ChatGPT |    63.9   |     **57.0**    |  44.7   | 55.2 |
|      LLaMA 7B   |    5.2   |     27.1    |  24.3   | 18.9 |
|      LLaMA 33B |    1.8   |     43.4    |  30.3   | 25.2 |
|   |      |             |            |    |  |
|   Task-tuned LLaMA 7B (Full) |   **75.1**   |     44.5    |  49.9   | 56.5 |
| Task-tuned LLaMA 33B (LoRA) |  74.0  |  51.3   | **50.2**|**58.5**|

The LLaMA 33B (LoRA) performance is achieved with only **~16h** finetuning on the training split of PubMedQA and MedMCQA with a single 8 \* A100 server. 
For more performance, including instruction tuning results, please refer to our [Documentation](https://optimalscale.github.io/LMFlow/).

## Model Zoo
We open-sourced the trained checkpoints to everyone for further training and inference.

| Instruct-tuned Models   |  Status | Base Model | Download | 
|----------|:-------------:|----------|:-------------:|
| Hu (湖羊) | ![completed](https://geps.dev/progress/100) | LLaMA-7B | [Google Drive](https://drive.google.com/file/d/1x5JLae3akVkfFeDhSe3TEyUbPn_GNFyb/view?usp=share_link) |
| Dongshan (东山羊) | ![completed](https://geps.dev/progress/100) | LLaMA-13B |  [Google Drive](https://drive.google.com/file/d/1m_rpe6rNpN59kWvjJ3GfKeEmS-68TRYr/view?usp=share_link) |
| Hetian (和田羊) | ![completed](https://geps.dev/progress/100) |LLaMA-33B |  [Google Drive](https://drive.google.com/file/d/1IqgqLHwNkWQ7BffheZnqD6a-8Zul1bk6/view?usp=share_link) |
| Altay (阿勒泰羊) | ![training](https://geps.dev/progress/65) | LLaMA-65B | Google Drive |
| LLaMA7B-medical | ![completed](https://geps.dev/progress/100) | LLaMA-7B | [Google Drive](https://drive.google.com/file/d/1Z44tsrRvfDFvucbNGFjHC_vbPcBvg3x-/view?usp=share_link) |
| LLaMA13B-medical | ![completed](https://geps.dev/progress/100) | LLaMA-13B |  [Google Drive](https://drive.google.com/file/d/1uoTAXTMyYQkP6N4ummx7tj-c4v1p91ap/view?usp=share_link) |
| LLaMA33B-medical | ![completed](https://geps.dev/progress/100) |LLaMA-33B |  [Google Drive](https://drive.google.com/file/d/14N9o_1pwHmVuSikQ3orMVzZDrLYJC0iM/view?usp=share_link) |
| LLaMA65B-medical | ![training](https://geps.dev/progress/90) | LLaMA-65B | Google Drive |


## Supported Pipelines

| Pipelines   |   Status |
|----------|:-------------:|
| Task Tuning |  :white_check_mark: Supported |
| Instruction Tuning |  :white_check_mark: Supported |
| Parameter-Efficient Tuning |  :white_check_mark: Supported |
| Large Model Inference |  :white_check_mark: Supported |
| Reinforced Tuning |  :wrench: Developing |


## Supported Models
Seamlessly supported the models in 🤗 huggingface.

| Models   |  Status | |  Models | Status | 
|----------|:-------------:|----------|----------|:-------------:|
| GPT2-large |  :white_check_mark: Tested | | Galactica-6.7B |  :wrench: Untested |
| GPT2-xl |  :white_check_mark: Tested | | Galactica-30B |  :wrench: Untested |
| GPT-Neo-1.3B |  :wrench: Untested | | LLaMA-7B |  :white_check_mark: Tested :star: |
| GPT-Neo-2.7B |  :wrench: Untested | | LLaMA-13B |  :white_check_mark: Tested :star: |
| GPT-Neox-20B |  :wrench: Untested | | LLaMA-33B |  :white_check_mark: Tested :star: |
| Galactica-1.3B |  :white_check_mark: Tested | |LLaMA-65B |  :wrench: Untested |

## 1.Setup
```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```

## 2.Prepare Dataset
You can easily download the example training dataset and test dataset by running 
```bash
cd data
bash download.sh all
cd -
``` 
If you cannot access Google Drive, you can download the data by [BaiduNetDisk](https://pan.baidu.com/s/1L7AC5Oy-3YhbCp2aNX4tnQ?pwd=dm2s).

You can also use your own dataset by simply convert to the following format:
```json
{
  "type": "text2text",
  "instances": [
    {
      "input": "Question: The Transformer architecture [START_REF]",
      "output": "N/A"
    },
    ...
  ]
}
```
```json
{
  "type": "text_only",
  "instances": [
    {
      "text": "Defintion: In this task, we ask you to write an answer to a question that involves events that may be stationary (not changing over time) or transient (changing over time). For example, the sentence \"he was born in the U.S.\" contains a stationary event since it will last forever; however, \"he is hungry\" contains a transient event since it will remain true for a short period of time. Note that a lot of the questions could have more than one correct answer. We only need a single most-likely answer. Please try to keep your \"answer\" as simple as possible. Concise and simple \"answer\" is preferred over those complex and verbose ones. \n Input: Question: Sentence: It's hail crackled across the comm, and Tara spun to retake her seat at the helm. \nQuestion: Will the hail storm ever end? \n Output: NA \n\n"
    },
    ...
  ]
}
```
## 3. Run Scripts
### 3.1 Run Finetuning

You can run `scripts/run_finetune.sh` to finetune a GPT-2 base model
```sh
./scripts/run_finetune.sh
```

If you would like to provide arguments for deepspeed to reflect your machine
settings, you may pass the corresponding deepspeed arguments to the script. For
example,
```sh
./scripts/run_finetune.sh "--num_gpus=8 --master_port 10001"
```

To enable LoRA finetuning, you may refer to
```sh
./scripts/run_finetune_with_lora.sh
```
which can be run in similar manner.

For detailed configurations, one may modify these scripts directly. These
scripts actually just call python script `examples/finetune.py`, which can
be run in following manner,

```sh
deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 \
    --run_name finetune_with_lora \
    --model_name_or_path facebook/galactica-1.3b \
    --num_train_epochs 0.01 \
    --learning_rate 2e-5 \
    --dataset_path ${dataset_path} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --block_size 512 \
    --do_train \
    --output_dir output_models/finetune \
    --overwrite_output_dir \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1
```
Here we set number of epochs `--num_train_epochs` to `0.01` so that the
finetuning process can be finished quickly. If you wish to obtain a model with
better performance, feel free to adjust those hyperparameters. You may run
```python
python examples/finetune.py -h
```
to view all possible finetuning arguments. The finetuned model checkpoint will
be saved in the argument specified by `--output_dir`, which is
`output_models/finetune` in the above example.
### 3.2 Run Evaluation

One can directly run evaluation with an existing huggingface model, e.g. to run
GPT2 large, one may execute
```sh
./scripts/run_evaluation.sh
```
or run the corresponding python script
```python
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --test_file data/MedQA-USMLE/validation/valid_1273.json \
    --deepspeed examples/ds_config.json \
```
To load the finetuned model, specify `--model_name_or_path` with the saved
model checkpoint directory path.

For LoRA finetuned models, one may refer to
```sh
./scripts/run_evaluation_with_lora.sh
```

Those scripts invoke the examples `examples/*.py` built based on our APIs. For
more API-related examples, one may refer to the methods in the unittest
`tests`.

## 4. Additional Notes
### 4.1 LLaMA Checkpoint

1. First, you need to get the access of LLaMA model from [facebookresearch/llama](https://github.com/facebookresearch/llama). Download the official checkpoints and save them into `${llama-path}`.

2. Second, convert the official checkpoints `${llama-path}` to HuggingFace supported checkpoints `${llama-hf-path}` by running

    `python ./scripts/convert_llama_weights_to_hf.py --input_dir ${llama-path} --model_size 7B --output_dir ${llama-hf-path}/llama-7b-hf`

3. Then you are good to go by setting the checkpoint path to `${llama-hf-path}/llama-7b-hf`. Enjoy it!

4. (optional) Now you have the original llama-7b-hf pretrained model. With
```sh
cd output_models && ./download.sh all && cd -
```
You can obtain the model difference finetuned by ours. By a way similar to `./scripts/run_evaluation_with_lora.sh`,
```sh
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text \
    --model_name_or_path ${llama-hf-path}/llama-7b-hf \
    --lora_model_path output_models/${llama-model-diff-path} \
    --test_file data/alpaca/test/test_252.json \
    --deepspeed examples/ds_config.json
```
You can now evaluate with the finetuned llama model.

### 4.2 DeepSpeed Config
You can config the deepspeed under configs. Details can be referred at [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)

## 5. Model Release

### 5.1 Medical Model Checkpoints
You can run following script to download our medical model checkpoints :

```bash
cd output_models
bash download.sh medical_ckpt
cd -
```
You can also directly download our model via google drive link : [medical_ckpt.tar.gz](https://drive.google.com/file/d/1bnsQGNGNYchsOfiNyRAmL2fNiowbmFNw/view?usp=share_link)

### 5.2 Instruction Model Checkpoints
Similarly, you can run following script to download our instruction model checkpoints :
```bash
cd output_models
bash download.sh instruction_ckpt
cd -
```

You can also directly download our model via google drive link : [instruction_ckpt.tar.gz](https://drive.google.com/file/d/1d_ioQ-ViVweeifbsFSO4pczc3UORFHZO/view?usp=share_link)

### 5.3 Begin Reproduce

After downloading the model checkpoints. You can replace the `--lora_model_path` with `output_models/instruction_ckpt/llama7b-lora` (example for llama-7b for instruction) and replace `--model_name_or_path` with your converted llama model inside `LMFlow/scripts/run_evaluation_with_lora.sh` and run this shell script to reproduce the result.

Then you can check the model performance at our [Doc](https://optimalscale.github.io/LMFlow/).

## Documentation
Please refer to our [Documentation](https://optimalscale.github.io/LMFlow/) for more API reference and experimental results.

## Vision
Hello there! We are excited to announce the upcoming release of our code repository that includes a complete LLM training process, enabling users to quickly build their own language models and train them effectively.

Our code repository is not just a simple model; it includes the complete training workflow, model optimization, and testing tools. You can use it to build various types of language models, including conversation models, question-answering models, and text generation models, among others.

Moreover, we aim to create an open and democratic LLM sharing platform where people can share their checkpoints and experiences to collectively improve the skills of the community. We welcome anyone who is interested in LLM to participate and join us in building an open and friendly community!

Whether you are a beginner or an expert, we believe that you can benefit from this platform. Let's work together to build a vibrant and innovative LLM community!

我们很高兴地开源LMFlow代码库，其中包括了完整的大模型训练流程，能够快速、高效地训练和部署自己的语言模型。

我们的代码库不仅仅是一个简单的模型； 它包括完整的训练流程、模型权重和测试工具。 您可以使用它来构建各种类型的语言模型，包括对话模型、问答模型和文本生成模型等。

此外，我们旨在创建一个开放和民主的大模型共享平台，任何人都可以在这个平台上分享训练模型权重和经验。 我们欢迎任何对大模型感兴趣的人参与进来，与我们一起建设一个开放友好的社区！

无论您是初学者还是专家，我们相信大家都能从这个平台中获益。让我们共同努力，建立一个充满活力和创新的大模型社区！

[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/srGxyazbNs)
[![WeChat badge](https://img.shields.io/badge/微信-加入-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/02/iHtF23.png)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)

## Disclaimer

This package aims to provide a streamlined and user-friendly pipeline for large model tuning. Its functionalities serve as a reference and are intended for use by the user. However, it is important to note that the responsibility for the preparation of the data and pretrained models lies solely with the user. This package does not guarantee the accuracy, completeness, applicability, or legality of the components from the user's preparation. Users must be aware of and assume all risks and liabilities associated with the preparation of the models and data, and obtain legal, commercial, and technical advice before utilizing this package. The pipeline shall not be held responsible for any direct, indirect, special, incidental, or consequential damages resulting from the user's improper preparation of the data and pretrained models.   

It is also crucial to highlight that the results generated by the model are based on probabilistic models and not directly related to this pipeline. The accuracy, reliability, applicability, and legality of the results are not guaranteed by this pipeline. Therefore, users must also be aware of the risks and liabilities associated with the results and seek legal, commercial, and technical advice before relying on the model-generated outcomes. This pipeline shall not be accountable for any direct, indirect, special, incidental, or consequential damages resulting from the user's reliance on the model-generated results.

## Support

If you need any help, please submit a [Github](https://github.com/OptimalScale/LMFlow) issue.

## Contributors
<a href="https://github.com/OptimalScale/LMFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OptimalScale/LMFlow" />
</a>


## Citation
If you find this repository useful, please consider giving ⭐ and citing:

```
@misc{lmflow,
  author = {Shizhe Diao and Rui Pan and Hanze Dong and KaShun Shum and Jipeng Zhang and Wei Xiong and Tong Zhang},
  title = {LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://optimalscale.github.io/LMFlow/}},
}
```
