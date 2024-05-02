<p align="center" width="100%">
<img src="../assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README.md">English</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_es.md">Español</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_jp.md">日本語</a> |
        <b>한국어</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_hindi.md">हिंदी</a>
    <p>
</h4>

> [!NOTE]
> The Korean README file was translated by LLM for reference only. Korean speakers are welcome to submit a PR to polish the document!  

> [!NOTE]  
> 한국어 README 파일은 참고용으로 LLM에 의해 번역되었습니다. 한국어 사용자들은 문서를 개선하기 위해 PR을 제출할 것을 환영합니다!  

[![Website](https://img.shields.io/badge/Website-Demo-20B2AA.svg)](https://lmflow.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/Discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://ibb.co/ZhM4hhn)

다음은 사용자 친화적이고 빠르며 신뢰할 수 있으며 커뮤니티 전체에 액세스할 수 있도록 설계된 대규모 기계 학습 모델을 미세 조정하는 데 유용한 확장 가능하고 편리하며 효율적인 도구 상자입니다.

<p align="center" width="100%">
<img src="../assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Latest News
* [2024-04-25] :rocket: 대화 템플릿을 지원합니다! 최신 [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B) 및 [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) 대화 템플릿과 `chatml`과 같은 자주 사용되는 템플릿을 미리 설정해 두었습니다 ([여기](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#conversation-template)에서 모든 템플릿을 확인하세요). 더 많은 미리 설정된 템플릿을 추가하는 작업 중에 있습니다. 셸 스크립트에 해당하는 `--conversation_template`를 추가하면 됩니다! :rocket:
* [2024-03-27] [LISA](https://arxiv.org/abs/2403.17919)를 지원합니다. 메모리를 비우지 않고도 24G 메모리에서 7B 훈련이 가능합니다!  
* [2023-09-11] [추론적 디코딩 (speculative decoding)](https://arxiv.org/abs/2211.17192)을 지원합니다. 사용법 및 가속화 세부 정보는 [speculative_decoding](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) 를 확인하세요.
* [2023-08-14] LLaMA 모델에 대한 위치 보간(선형 및 NTK 스케일링)을 사용하여 긴 문맥 추론을 지원합니다. 자세한 내용은 [Postion Interpolation](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md) 를 확인하세요.
* [2023-08-07] [Flash Attention-2](https://crfm.stanford.edu/2023/07/17/flash2.html)를 지원합니다. 자세한 내용은 [Flash Attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) 를 확인하세요.


## Table of Contents
- [LMFlow](#lmflow)
  - [Latest News](#latest-news)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Setup](#setup)
    - [Prepare Dataset](#prepare-dataset)
    - [Fine-Tuning (Full)](#fine-tuning-full)
    - [Fine-Tuning (LISA)](#fine-tuning-lisa)
    - [Fine-Tuning (LoRA)](#fine-tuning-lora)
    - [Inference](#inference)
    - [Deployment](#deployment)
    - [Evaluation](#evaluation)
  - [Supported Features](#supported-features)
  - [Support](#support)
  - [License](#license)
  - [Citation](#citation)


## Quick Start
### Setup
저희의 Repo는 이미 리눅스 (우분투 20.04)에서 완전한 테스트가 이루어졌습니다. 다른 운영 체제 플랫폼 (맥OS, 윈도우)은 아직 완전히 테스트되지 않았으므로 예상치 못한 오류가 발생할 수 있습니다. 먼저 리눅스/윈도우 WSL에서 사용해보거나 Google Colab을 사용하는 것을 권장합니다.
CUDA 10.3-11.7에 대해서는 `v0.0.5` 및 그 이전 버전을 사용하는 것이 좋습니다. 11.7보다 큰 CUDA의 경우, 더 나은 경험을 위해 우리의 stable 브랜치인 `>= v0.0.6` 을 사용하십시오.
```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```

### Prepare Dataset
저희의 [공식 문서(영문)](https://optimalscale.github.io/LMFlow/examples/DATASETS.html) 를 참고해 주세요. 공식 문서는 현재 번역 중이며, 조금만 기다려 주시기 바랍니다.

### Fine-Tuning (Full)
전체 매개변수 파인 튜닝은 모델의 모든 매개변수를 업데이트합니다. GPT-2의 전체 매개변수 파인 튜닝의 예시는 아래와 같습니다:

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

> [!TIP]
> 대화 데이터셋에 대화 템플릿을 지정하려면 `--conversation_template` 매개변수를 추가할 수 있습니다.
> 
> <details><summary>예시: Llama-3-8B에 대화 데이터셋 템플릿 지정</summary>  
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

### Fine-Tuning (LISA)
[LISA](https://arxiv.org/abs/2403.17919) 는 **메모리 효율적인(memory-efficient)** 파인 튜닝 알고리즘이며, 메모리와 무작위로 해동하는 레이어 수 사이의 균형을 가능하게 합니다. 아래 스크립트는 현재 **단일 GPU** 에서만 테스트되었습니다. 최신 업데이트에 주목해 주세요! :smile:
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
> <details><summary>예시: Llama-2-7B 대화 데이터셋 템플릿 지정</summary>  
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
></details>

### Fine-Tuning (LoRA)
LoRA는 전체 매개변수 미세 조정보다 더 효율적인 매개변수 효율적인 미세 조정 알고리즘입니다.
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path facebook/galactica-1.3b \
  --dataset_path data/alpaca/train_conversation \
  --output_lora_path output_models/finetuned_galactica_lora
```

> [!TIP]
> <details><summary>예시: Llama-2-7B 대화 데이터셋 템플릿 지정</summary>  
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
> <details><summary>LoRA 가중치 병합</summary>
>
>아래 명령어를 사용하여 LoRA 가중치를 원본 모델과 병합할 수 있습니다:  
>```sh
>./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>

### Inference
미세 조정이 완료된 후에는 다음 명령을 사용하여 모델과 대화할 수 있습니다.
```sh
./scripts/run_chatbot.sh output_models/finetuned_gpt2
```

### Deployment
지역에 모델을 배포하려는 경우, Gradio 기반의 챗봇 UI를 제공합니다. Robin-7b의 데모를 시작하려면 다음 명령을 참고하세요:
```sh
pip install gradio
python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path YOUR-LLAMA  --lora_model_path ./robin-7b --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:"       --end_string "#" --max_new_tokens 200
```

### Evaluation
[LMFlow Benchmark](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418) 은(는) 오픈 소스 LLM을 자동으로 평가하기 위한 프레임워크입니다. 우리는 Negative Log Likelihood (NLL)을 평가 지표로 사용하여 대화, 상식 추론 및 지시 따름 능력과 같은 LLM의 여러 측면을 평가합니다. LMFlow Benchmark를 사용하여 손에 있는 모델을 평가하고 [모델 비교 (LLM Comparision)](https://docs.google.com/spreadsheets/d/1JYh4_pxNzmNA9I0YM2epgRA7VXBIeIGS64gPJBg5NHA/edit?usp=sharing)에 참여하십시오.

GPT-2 XL을 예로 들면 다음 명령으로 평가를 시작할 수 있습니다:
```sh
./scripts/run_benchmark.sh --model_name_or_path gpt2-xl
```
`--model_name_or_path`은 필수 입력 항목이며, huggingface 모델 이름 또는 모델의 로컬 경로를 전달할 수 있습니다. `./output_dir/gpt2-xl_lmflow_chat_nll_eval`, `./output_dir/gpt2-xl_all_nll_eval`, 그리고 `./output_dir/gpt2-xl_commonsense_qa_eval` 폴더 내의 `benchmark.log`를 통해 평가 결과를 확인할 수 있습니다.


## Supported Features
<details> <summary>미세 조정 가속 & 메모리 최적화</summary>

* LISA: 메모리 효율적인 대규모 언어 모델 미세 조정을 위한 레이어별 중요도 샘플링

  LISA는 메모리 효율적인 LLM 미세 조정 알고리즘이다. 미세 조정 과정에서 층을 선택적으로 고정함으로써, LISA는 LoRA와 같은 기존의 미세 조정 방법을 뛰어넘는다. 자세한 내용은 [논문](https://arxiv.org/abs/2403.17919)을 참조하십시오.
  훈련 명령어에 `--use_lisa 1` 매개변수를 지정하여 LISA를 사용할 수 있습니다. 활성화된 층의 수는 `--lisa_activated_layers 2`로 제어되며, 고정된 층의 간격은 `--lisa_step_interval 20`으로 조정할 수 있습니다.

* LoRA

  LoRA는 전체 파라미터 튜닝보다 효율적인 파라미터 효율적인(feasible-efficient) 튜닝 알고리즘입니다. 자세한 내용은 [Fine-tuning (LoRA)](#Fine-tuning-LoRA)를 참조하십시오.

* FlashAttention

  FlashAttention-1 및 FlashAttention-2를 지원합니다. 자세한 내용은 [FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)를 참조하십시오.

* Gradient Checkpointing

  [Gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)은 메모리 최적화 기술로, 핵심 아이디어는 메모리 점유를 줄이기 위해 계산을 메모리와 교환하는 것입니다. 훈련 명령에 `--gradient_checkpointing`을 추가하여 사용할 수 있습니다.

* Deepspeed Zero3

  LMFlow는 [Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)를 지원합니다. 사용 가능한 [deepspeed 설정 파일](https://github.com/OptimalScale/LMFlow/blob/main/configs/ds_config_zero3.json)을 제공합니다.

</details>


<details> <summary>추론 가속화</summary>

* LLaMA CPU 추론
  
  [llama.cpp](https://github.com/ggerganov/llama.cpp)에 감사드립니다. 이제 모든 사람이 CPU에서 자신의 LLaMA(4-bit 양자화)를 실행할 수 있습니다! 우리는 LLaMA LoRA 가중치를 `.pt` 파일로 변환하는 스크립트를 제공하며, llama.cpp의 `convert-pth-to-ggml.py`를 사용하여 모델 양자화를 수행하여 LLaMA CPU 추론을 진행할 수 있습니다.

* FlashAttention

  FlashAttention-1 및 FlashAttention-2를 지원합니다. 자세한 내용은 [FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)를 참조하십시오.

</details>


<details> <summary>긴 텍스트</summary>

* LLaMA 모델의 위치 보간 (Position Interpolation)
  
  위치 보간 (Linear & NTK scaling을 통한)을 지원하여 LLaMA의 컨텍스트 창을 확장합니다. 자세한 내용은 여기를 참조하세요: [위치 보간](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md)。

</details>


<details> <summary>모델 커스터마이징</summary>

* 어휘 확장
  
  자체 sentencepiece tokenizer를 학습한 다음 모델에 내장된 huggingface tokenizer와 결합하세요! 자세한 내용은 여기를 참조하세요: [어휘 확장](https://github.com/OptimalScale/LMFlow/blob/main/scripts/vocab_extension)。

</details>


<details> <summary>다중 모달</summary>

* 다중 모달 챗봇
  
  LMFlow는 다중 모달 (이미지, 텍스트) 입력을 지원합니다. 자세한 내용은 여기를 참조하세요: [LMFlow 다중 모달 챗봇](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_vis_chatbot_gradio_minigpt4.sh)。

</details>


## Support
도움이 필요하면 공식 [깃 허브 레포지토리](https://github.com/OptimalScale/LMFlow)에 이슈를 생성해주세요.


## License
이 프로젝트에 포함된 코드는 Apache 2.0 라이센스를 사용합니다. 이 프로젝트에 포함된 모델을 상업적 용도로 사용하려는 경우, 프로젝트 개발자에게 허가를 요청하십시오. 


## Citation
이 repository를 유용하게 사용하셨다면 ⭐을 눌러주시고 다음을 통해 인용해주시면 감사하겠습니다. [arXiv](https://arxiv.org/abs/2306.12420)

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
