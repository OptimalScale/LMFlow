<p align="center" width="100%">
<img src="../assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README.md">English</a> |
        <b>简体中文</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_es.md">Español</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_jp.md">日本語</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_ko.md">한국어</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_hindi.md">हिंदी</a>
    <p>
</h4>

[![Website](https://img.shields.io/badge/Website-Demo-20B2AA.svg)](https://lmflow.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/Discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://ibb.co/ZhM4hhn)

一个可扩展、方便和高效的工具箱，用于微调大型机器学习模型。我们的目标是开发一套用户友好、快速可靠，并对整个社区开放的全流程微调代码库。

<p align="center" width="100%">
<img src="../assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## 新闻
* [2024-04-25] :rocket: 支持多轮对话数据格式以及对话模板！我们已经添加了近期热门模型 [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B) 和 [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)的对应模板，也提供了一些如`chatml`等常用的模板（[这里](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#conversation-template)查看所有已经预设的模板），更多模板正在添加中。在微调shell脚本里指定对应的`--conversation_template`试试吧！ :rocket:  
* [2024-03-27] 支持 [LISA](https://arxiv.org/abs/2403.17919) —— 无需offloading，在24G显存的GPU上训练7B模型！  
* [2023-09-11] 支持 [投机解码(speculative decoding)](https://arxiv.org/abs/2211.17192)， 点击 [使用指南](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) 查看使用方法和简单的性能统计。
* [2023-08-14] 支持通过位置插值（Postion Interpolation）（Linear & NTK scaling）扩展LLaMA的上下文窗口，查看详情：[位置插值](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md)。
* [2023-08-07] 支持 [Flash Attention-2](https://crfm.stanford.edu/2023/07/17/flash2.html)，查看详情：[Flash Attention使用指南](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)。
* [2023-08-02] 支持 [Llama2](https://ai.meta.com/llama/)，[ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b)，[Baichuan](https://huggingface.co/baichuan-inc/Baichuan-7B)。


## 目录
- [LMFlow](#lmflow)
  - [新闻](#新闻)
  - [目录](#目录)
  - [快速上手](#快速上手)
    - [安装](#安装)
    - [准备数据集](#准备数据集)
    - [微调（全参数）](#微调全参数)
    - [微调（LISA）](#微调lisa)
    - [微调（LoRA）](#微调lora)
    - [推理](#推理)
    - [部署](#部署)
    - [评测](#评测)
  - [支持功能](#支持功能)
  - [需要帮助？](#需要帮助)
  - [协议](#协议)
  - [引用](#引用)


## 快速上手
### 安装
我们的Repo已经在Linux（Ubuntu 20.04）上进行了测试。其他操作系统平台（MacOS、Windows）尚未完全测试，因此可能会遇到一些预期外的错误。建议先在Linux/Windows WSL上尝试使用，或者使用Google Colab来体验。

对于CUDA 10.3-11.7，建议使用`v0.0.5`及更早版本。对于大于11.7的CUDA，请使用我们的稳定分支`>= v0.0.6`以获得更好的体验。
```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```

### 准备数据集
请参考我们的 [官方文档（英文版）](https://optimalscale.github.io/LMFlow/examples/DATASETS.html)。官方文档正在汉化中，请耐心等待。

### 微调（全参数）
全参数微调将更新模型的所有参数。全参数微调GPT-2的示例如下：

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

> [!TIP]
> 可以通过添加`--conversation_template`参数为对话数据集指定对话模板。
> 
> <details><summary>示例：为 Llama-3-8B 指定对话数据集模板</summary>  
> 
>```bash
>cd data && ./download.sh alpaca && cd -
>
>./scripts/run_finetune.sh \
>  --model_name_or_path meta-llama/Meta-Llama-3-8B \
>  --dataset_path data/alpaca/train_conversation \
>  --conversation_template llama3 \
>  --output_model_path output_models/finetuned_llama3_8b
>```
> </details>

### 微调（LISA）
[LISA](https://arxiv.org/abs/2403.17919) 是一种 **内存高效（memory-efficient）** 的微调算法，它允许在内存和随机解冻的层数之间进行权衡。下面的脚本目前仅在 **单个GPU** 上进行了测试。请关注我们的最新更新！ :smile:
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lisa.sh \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_llama2_7b \
  --lisa_activated_layers 1 \
  --lisa_interval_steps 20
```

> [!TIP]
> <details><summary>示例：为 Llama-2-7B 指定对话数据集模板</summary>  
> 
>```bash
>cd data && ./download.sh alpaca && cd -
>
>./scripts/run_finetune_with_lisa.sh \
>  --model_name_or_path meta-llama/Llama-2-7b-hf \
>  --dataset_path data/alpaca/train_conversation \
>  --conversation_template llama2 \
>  --output_model_path output_models/finetuned_llama2_7b_lisa \
>  --lisa_activated_layers 1 \
>  --lisa_interval_steps 20
>```
> </details>

### 微调（LoRA）
LoRA 是一种比全参数微调更为高效的 **参数高效（parameter-efficient）** 微调算法。
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path facebook/galactica-1.3b \
  --dataset_path data/alpaca/train_conversation \
  --output_lora_path output_models/finetuned_galactica_lora
```

> [!TIP]
> <details><summary>示例：为 Llama-2-7B 指定对话数据集模板</summary>  
> 
>```bash
>cd data && ./download.sh alpaca && cd -
>
>./scripts/run_finetune_with_lora.sh \
>  --model_name_or_path meta-llama/Llama-2-7b-hf \
>  --dataset_path data/alpaca/train_conversation \
>  --conversation_template llama2 \
>  --output_model_path output_models/finetuned_llama2_7b_lora \
>```
> </details>
>
> <details><summary>合并LoRA权重</summary>
>
>可以通过下面的指令把LoRA权重和原模型合并:  
>```sh
>./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>

### 推理
在微调结束后，可以通过以下命令与模型进行对话。
```sh
./scripts/run_chatbot.sh output_models/finetuned_gpt2
```

### 部署
如果您想在本地部署自己的模型，我们提供了基于gradio的聊天机器人UI。
以下命令可以启动robin-7b的demo，请参考：
```sh
pip install gradio
python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path YOUR-LLAMA  --lora_model_path ./robin-7b --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:"       --end_string "#" --max_new_tokens 200
```

### 评测
[LMFlow Benchmark](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418) 是一个针对开源LLM的自动评估框架。我们使用Negative Log Likelihood (NLL)作为指标来评估LLM的各个方面，如：闲聊、常识推理和指令遵循能力。欢迎使用LMFlow Benchmark对您手上的模型进行评测，并参与我们的 [模型比较（LLM comparision）](https://docs.google.com/spreadsheets/d/1JYh4_pxNzmNA9I0YM2epgRA7VXBIeIGS64gPJBg5NHA/edit?usp=sharing)。

以GPT-2 XL为例，通过以下指令开始评测：
```sh
./scripts/run_benchmark.sh --model_name_or_path gpt2-xl
```
`--model_name_or_path`是必填参数，可以传入 huggingface模型名 或 模型的本地路径。
可以通过`./output_dir/gpt2-xl_lmflow_chat_nll_eval`、`./output_dir/gpt2-xl_all_nll_eval` 和`./output_dir/gpt2-xl_commonsense_qa_eval`下的`benchmark.log`查看评测结果。


## 支持功能
<details> <summary>微调加速 & 内存优化</summary>

* LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning
  
  LISA是一种内存高效的LLM微调算法。通过在微调过程中选择性地冻结层，LISA超越了现有的微调方法（如LoRA）。欢迎查阅 [论文](https://arxiv.org/abs/2403.17919) 了解更多。
  可以在训练命令中指定参数 `--use_lisa 1` 来使用LISA。通过 `--lisa_activated_layers 2` 来控制激活的层的数量，并通过 `--lisa_step_interval 20` 来调整冻结的层的间隔。

* LoRA
  
  LoRA 是一种比全参数微调更为高效的参数高效（parameter-efficient）微调算法，请参考：[微调（LoRA）](#微调lora)。

* FlashAttention
  
  我们支持FlashAttention-1 和 FlashAttention-2。更多细节见：[FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)。

* Gradient Checkpointing
  
  [Gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing) 是一种内存优化技术，核心思想是通过计算换取内存，从而减少显存占用。在训练命令中添加 `--gradient_checkpointing` 即可使用。

* Deepspeed Zero3
  
  LMFlow 支持 [Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)。我们提供了开箱即用的 [deepspeed配置文件](https://github.com/OptimalScale/LMFlow/blob/main/configs/ds_config_zero3.json)。

</details>


<details> <summary>推理加速</summary>

* LLaMA CPU推理
  
  感谢 [llama.cpp](https://github.com/ggerganov/llama.cpp)，现在所有人都能在CPU上运行自己的LLaMA（4-bit量化）了！我们提供了将LLaMA LoRA权重转换成`.pt`文件的脚本，只需要使用 llama.cpp 的 `convert-pth-to-ggml.py` 进行模型量化即可进行LLaMA CPU推理。

* FlashAttention
  
  我们支持FlashAttention-1 和 FlashAttention-2。更多细节见：[FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)。

</details>


<details> <summary>长文本</summary>

* LLaMA模型的位置插值（Position Interpolation）
  
  支持通过位置插值（Postion Interpolation）（Linear & NTK scaling）扩展LLaMA的上下文窗口，查看详情：[位置插值](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md)。

</details>


<details> <summary>模型定制</summary>

* 词表扩充
  
  训练自己的sentencepiece tokenizer，然后和模型自带的huggingface tokenizer进行合并！请参考：[词表扩充](https://github.com/OptimalScale/LMFlow/blob/main/scripts/vocab_extension) 。

</details>


<details> <summary>多模态</summary>

* 多模态Chatbot
  
  LMFlow 支持多模态（图、文）输入。请参考：[LMFlow multimodal chatbot](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_vis_chatbot_gradio_minigpt4.sh)。

</details>


## 需要帮助？
如果您需要任何帮助，欢迎提交[Github Issue](https://github.com/OptimalScale/LMFlow/issues)。


## 协议
本项目所含代码采用Apache 2.0协议。如果您希望将本项目所含模型用于商业用途，请填写并签署[本文件](https://docs.google.com/forms/d/e/1FAIpQLSertnFbm2_aELsPMwOu_DhAu3p7bQgv8_MWSug7D80AyzPLhg/viewform?usp=pp_url)取得授权。


## 引用
如果您觉得我们的Repo有用，欢迎点赞⭐、fork、转发和引用我们的[论文](https://arxiv.org/abs/2306.12420)：

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