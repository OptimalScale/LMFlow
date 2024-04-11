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


## 최신 뉴스
* [2024-03-27] :rocket: [LISA](https://arxiv.org/abs/2403.17919)를 지원합니다. 메모리를 비우지 않고도 24G 메모리에서 7B 훈련이 가능합니다! :rocket:
* [2023-09-11] [추론적 디코딩 (speculative decoding)](https://arxiv.org/abs/2211.17192)을 지원합니다. 사용법 및 가속화 세부 정보는 [speculative_decoding](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) 를 확인하세요.
* [2023-08-14] LLaMA 모델에 대한 위치 보간(선형 및 NTK 스케일링)을 사용하여 긴 문맥 추론을 지원합니다. 자세한 내용은 [Postion Interpolation](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md) 를 확인하세요.
* [2023-08-07] [Flash Attention-2](https://crfm.stanford.edu/2023/07/17/flash2.html)를 지원합니다. 자세한 내용은 [Flash Attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) 를 확인하세요.
* [2023-08-02] [Llama2](https://ai.meta.com/llama/), [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b) 및 [Baichuan](https://huggingface.co/baichuan-inc/Baichuan-7B) 모델을 지원합니다.


## 빠른 시작
### 설치
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

### 데이터셋 준비
저희의 [공식 문서(영문)](https://optimalscale.github.io/LMFlow/examples/DATASETS.html) 를 참고해 주세요. 공식 문서는 현재 번역 중이며, 조금만 기다려 주시기 바랍니다.

### 파인 튜닝 (전체 매개변수)
> [!IMPORTANT]
> 최근에 데이터 저장 서버에 일부 문제가 발생했습니다. 데이터를 다운로드할 때, 최신 스크립트인 메인 브랜치의 [`download.sh`](https://github.com/OptimalScale/LMFlow/blob/main/data/download.sh) 를 사용해주시기 바랍니다. 불편을 끼쳐드려 죄송합니다.

전체 매개변수 파인 튜닝은 모델의 모든 매개변수를 업데이트합니다. GPT-2의 전체 매개변수 파인 튜닝의 예시는 아래와 같습니다:
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train \
  --output_model_path output_models/finetuned_gpt2
```

### Online Service
> LMflow의 [웹 서비스를](https://lmflow.com/) 방문해주시면 감사하겠습니다. LMflow의 웹사이트에 LLaMA-7B-tuned와 LLaMA-33B-tuned를 미리 배포해 놓았습니다.  웹사이트 트래픽이 많을 경우, 웹사이트가 적절하게 응답하지 않을 수 있지만, 웹 서비스의 `Local Deploy`를 참조하여 직접 배포해보실 수도 있습니다.

### Colab chatbot(shell)
<p align="center" width="100%">
<img src="../assets/colab-shell-chatbot-demo.png">
</p>

LMflow는 구글 코랩의 T4/P100/V100 GPU를 이용한 간단한 쉘 챗봇 데모를 제공합니다. 데모로 제공되는 `gpt-neo-2.7b` 모델은 영어로만 사용하실 수 있고, 다른 LLM 모델에 비해 성능이 뛰어나지 않은 데모용 모델임으로 참고만 해주시면 감사하겠습니다. 유저는 LMFlow을 통해 자신의 데이터셋에 모델을 파인튜닝하고 더 나은 성능을 얻을 수 있습니다. 또한, 🤗[huggingface](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)에서 제공하는 다른 decoder-only 모델들 또한 다음과 같이 파인튜닝 해보실 수 있습니다.

```sh
./scripts/run_chatbot.sh {another-model-name}
```




### Colab chatbot(web)
LMflow는 구글 코랩의 T4/P100/V100 GPU를 이용한 간단한 웹 데모 챗봇을 제공합니다. 데모로 제공되는 `gpt-neo-2.7b` 모델은 영어로만 사용하실 수 있고, 다른 LLM 모델에 비해 성능이 뛰어나지 않은 데모용 모델임으로 참고만 해주시면 감사하겠습니다.

### Local Deploy
충분한 로컬 리소스가 있고, 모델을 로컬에서 배포하고 싶어하는 유저를 위해, LMflow는 백엔드(다른 프론트엔드에 서비스를 제공하기 위해)와 interactive 웹 프론트엔드(직접 대화할 수 있게 해주는)의 launch를 위한 플라스크 서버를 쉽게 실행할 수 있는 방법을 제공합니다. 
```sh
cd ./service
python app.py
```

## Medical Performance
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


LLaMA 33B (LoRA)의 성능은 단일 8 \* A100 서버로 PubMedQA와 MedMCQA을 사용하여, 16시간동안 파인튜닝한 결과입니다. Instruction tuning 등을 포함해 더 많은 모델 성능에 대해서 알고 싶으시면, 다음 [문서](https://optimalscale.github.io/LMFlow/)를 참조해주세요.

## Model Zoo
LMflow는 학습된 체크포인트들을 통해 추가 학습 및 추론하실 수 있도록 모든 모델을 오픈소스로 제공합니다.

| Instruct-tuned Models   |  Status | Base Model | Download |
|----------|:-------------:|----------|:-------------:|
| LLaMA-7B-tuned | ![completed](https://geps.dev/progress/100) | LLaMA-7B | [Google Drive](https://drive.google.com/file/d/1x5JLae3akVkfFeDhSe3TEyUbPn_GNFyb/view?usp=share_link) |
| LLaMA-13B-tuned | ![completed](https://geps.dev/progress/100) | LLaMA-13B |  [Google Drive](https://drive.google.com/file/d/1m_rpe6rNpN59kWvjJ3GfKeEmS-68TRYr/view?usp=share_link) |
| LLaMA-33B-tuned | ![completed](https://geps.dev/progress/100) |LLaMA-33B |  [Google Drive](https://drive.google.com/file/d/1IqgqLHwNkWQ7BffheZnqD6a-8Zul1bk6/view?usp=share_link) |
| LLaMA-65B-tuned | ![training](https://geps.dev/progress/65) | LLaMA-65B | Google Drive |
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
| Alignment Tuning |  :wrench: Developing |



## Supported Models

🤗 huggingface의 모든 [디코더 모델](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)에 LMflow를 원활하게 적용해보실 수 있습니다. LLaMA, GPT2, GPT-Neo, Galactica 등은 완벽하게 테스트 완료되었습니다. 추후 LMflow는 인코더 모델도 지원할 예정입니다.


## 1.Setup

소프트웨어 패키지는 Linux 운영 체제(Ubuntu 20.04)에서 완벽하게 테스트되었습니다. 다른 운영 체제 플랫폼(MacOS, Windows)은 테스트를 진행하고 있으므로 예상치 못한 오류가 발생할 수 있습니다. Linux 시스템에서 사용하시는 것을 추천드리고, Google Colab을 통해 테스트해보시기 바랍니다.

```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```

## 2.Prepare Dataset
다음을 실행하면 예제 학습 데이터셋과 테스트 데이터셋을 쉽게 다운로드하실 수 있습니다.
```bash
cd data
bash download.sh all
cd -
```

다음 형식으로 간단히 변환하면 자신의 데이터셋을 모델학습에 사용할 수 있습니다.
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
### 3.1 Run 파인튜닝

`scripts/run_finetune.sh` 를 실행하여 GPT-2 베이스 모델을 파인튜닝할 수 있습니다.
```sh
./scripts/run_finetune.sh
```

deepspeed에 arguments를 추가로 입력하시고자 할 경우,
```sh
./scripts/run_finetune.sh "--num_gpus=8 --master_port 10001"
```

LoRA 파인튜닝을 활성하시고자 할 경우,
```sh
./scripts/run_finetune_with_lora.sh
```



자세한 설정은 이 스크립트들을 직접 수정할 수 있습니다. 이 스크립트들은 실제로는 파이썬 스크립트 `examples/finetune.py`를 호출하며, 다음과 같이 사용하실 수 있습니다.

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
여기서는 `--num_train_epochs`의 epoch 수를 `0.01` 로 설정하여 파인튜닝 프로세스를 빠르게 완료할 수 있습니다. 더 나은 성능의 모델을 얻고 싶다면 하이퍼파라미터를 조정하시면 됩니다. 


모든 가능한 파인튜닝 arguments를 확인하시려면,
```python
python examples/finetune.py -h
```

참고로 훈련 데이터 세트가 작은 경우 ``block_size`` 의 값을 낮춰야만 합니다. 그렇지 않으면 Epoch Iteration에서 샘플을 사용할 수 없게됩니다.

파인튜닝 된 모델 체크포인트는 위의 예시에서 `--output_dir`로 지정된 인자에 저장됩니다. 
이 경우에는`output_models/finetune` 입니다.

### 3.2 Run Evaluation

기존 huggingface 모델로 직접 Evaluation을 실행할 수 있습니다.

예를 들어 GPT2 large를 실행하려면, 다음을 쉘 스크립트를 사용하시거나
```sh
./scripts/run_evaluation.sh
```
다음 명령어를 통해 파이썬 스크립트를 실행하십시오.
```python
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --dataset_path data/MedQA-USMLE/validation \
    --deepspeed examples/ds_config.json
```
파인튜닝 된 모델을 로드하려면 저장된 모델 체크포인트 디렉토리 경로를 `--model_name_or_path`를 사용해 지정하십시오.

LoRA 파인튜닝 된 모델의 경우 다음을 참조하십시오.
```sh
./scripts/run_evaluation_with_lora.sh
```

이러한 스크립트는 저희의 API를 기반으로 구축된 예제 `examples/*.py` 를 호출합니다. 
더 많은 API 관련 예제는 unittest의 `tests` 메소드를 참조하십시오.

## 4. Additional Notes
### 4.1 LLaMA Checkpoint

1. 먼저 [facebookresearch/llama](https://github.com/facebookresearch/llama)에서 LLaMA 모델에 대한 액세스 권한을 얻어야합니다. 공식 체크포인트를 다운로드하고 경로를 `${llama-path}`에 저장하십시오.

2. 아래의 커맨드를 실행하여 공식 체크포인트 `${llama-path}`를 HuggingFace가 지원하는 체크포인트 `${llama-hf-path}`로 변환하십시오.

    `python ./scripts/convert_llama_weights_to_hf.py --input_dir ${llama-path} --model_size 7B --output_dir ${llama-hf-path}/llama-7b-hf`

3. 그 다음에 `${llama-hf-path}/llama-7b-hf`로 체크포인트 경로를 설정하시면 끝입니다.

4. (선택 사항) 오리지널 llama-7b-hf Pre-trained 모델 전부를 다운로드하시고자 할 경우 다음을 실행하세요.
```sh
cd output_models && ./download.sh all && cd -
```
`./scripts/run_evaluation_with_lora.sh`와 유사한 방식으로 다음을 실행하여, 파인튜닝한 모델 difference를 얻을 수 있습니다.
```sh
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text \
    --model_name_or_path ${llama-hf-path}/llama-7b-hf \
    --lora_model_path output_models/${llama-model-diff-path} \
    --dataset_path data/alpaca/test \
    --prompt_structure "Input: {input}" \
    --deepspeed examples/ds_config.json
```
이제 파인튜닝된 llama 모델로 평가할 수 있습니다.

### 4.2 DeepSpeed Config
config는 configs를 통해 구성할 수 있습니다. 자세한 내용은 [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)참조하십시오.

## 5. Model Release

### 5.1 Medical Model Checkpoints
다음 스크립트를 실행하여 저희의 의료 모델 체크포인트를 다운로드 하실 수 있습니다.

```bash
cd output_models
bash download.sh medical_ckpt
cd -
```
또한 다음 Google 드라이브 링크를 통해 직접 모델을 다운로드 할 수 있습니다 : [medical_ckpt.tar.gz](https://drive.google.com/file/d/1bnsQGNGNYchsOfiNyRAmL2fNiowbmFNw/view?usp=share_link)

### 5.2 Instruction Model Checkpoints
다음 스크립트를 실행하여 저희의 instruction 모델 체크포인트를 다운로드 하실 수 있습니다.
```bash
cd output_models
bash download.sh instruction_ckpt
cd -
```

다음 스크립트를 실행하여 저희의 instruction 모델 체크포인트를 다운로드 하실 수 있습니다. [instruction_ckpt.tar.gz](https://drive.google.com/file/d/1d_ioQ-ViVweeifbsFSO4pczc3UORFHZO/view?usp=share_link)

### 5.3 Begin Reproduce

모델 체크포인트를 다운로드 한 후에는 `--lora_model_path` 를 `output_models/instruction_ckpt/llama7b-lora`(llama-7b for instruction의 예시)로 대체하고, `--model_name_or_path` 를 `LMFlow/scripts/run_evaluation_with_lora.sh` 내부의 변환 된 llama 모델로 대체 한 다음 이 셸 스크립트를 실행하여 결과를 재현 할 수 있습니다.

그런 다음 [문서](https://optimalscale.github.io/LMFlow/)에서 모델 성능을 확인하실 수 있습니다.

## Documentation
더 많은 API 참조 및 실험 결과는 [Documentation](https://optimalscale.github.io/LMFlow/)를 참조하십시오.

## Vision
안녕하세요! LMflow는 완전한 LLM 학습 프로세스를 포함하여 사용자가 자신의 언어 모델을 빠르게 구축하고 효과적으로 학습 할 수 있도록하는 코드 repository가 곧 출시 될 것을 발표하게 되어 기쁩니다.

우리의 코드 repository는 단순한 모델이 아니며 완전한 학습 워크 플로, 모델 최적화 및 테스트 도구를 포함합니다. 대화 모델, 질문/답변 모델 및 기타 텍스트 생성 모델을 비롯한 다양한 유형의 언어 모델을 구축하는 데 사용할 수 있습니다.

또한 LMflow는 LLM 공유 플랫폼을 만들어 사람들이 체크포인트와 경험을 공유하여 커뮤니티의 기술을 함께 개선할 수 있는 개방적이고 민주적인 LLM 공유 플랫폼을 만들고자합니다. LLM에 관심있는 누구나 참여하여 친근하고 개방적인 커뮤니티를 만들어 가고자 합니다.

초보자든 전문가든 상관없이 이 플랫폼에서 많은 혜택을 받을 수 있을 것이라고 믿습니다. 함께 활기차고 혁신적인 LLM 커뮤니티를 만들어 봅시다!

[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)

## Disclaimer

이 패키지는 대형 모델 튜닝을 위한 간소화 된 사용자 친화적인 파이프 라인을 제공하는 것을 목표로합니다. 따라서 어떠한 법적인 책임도 지지 않습니다.
그 기능은 참조 용도로 제공되며 사용자가 사용하도록 의도되었습니다. 그러나 데이터 및 사전 학습된 모델과 관련된 책임은 사용자에게 달려있습니다. 이 패키지는 사용자 준비 구성 요소의 정확성, 완전성, 적용 가능성 또는 법적 적합성을 보증하지 않습니다. 사용자는 모델 및 데이터의 준비와 관련된 모든 위험과 책임을 인식하고 가정하고 이 패키지를 활용하기 전에 법적, 상업적 및 기술적 자문을 받아야만 합니다. 파이프 라인은 사용자의 잘못된 데이터 및 사전 학습 된 모델의 준비로 인한 어떠한 직접적인, 간접적인, 특수, 부수적 또는 결과적 손해에 대해서도 책임을 지지 않습니다.

영어와 중국어 버전 모두를 포함하는 점검 포인트는 연구 목적으로만 제공됩니다. 
이러한 체크 포인트에 포함 된 교육 데이터에는 ChatGPT 언어 모델에서 생성 된 결과가 포함됩니다. 이러한 체크 포인트의 배포 또는 사용을 보증하거나 장려하지 않습니다. 이러한 체크 포인트의 사용자는 올바르고 적절하게 사용되었는지 확인하는 것은 전적으로 사용자의 책임입니다.

또한 모델에서 생성 된 결과는 확률 모델에 기반하며 직접적으로 이 파이프 라인과 관련이 없음을 강조하는 것이 중요합니다. 결과의 정확성, 신뢰성, 적용 가능성 및 법적 적합성은 이 파이프 라인에서 보증되지 않습니다. 따라서 사용자는 결과와 관련된 위험과 책임도 인식해야하며 모델에서 생성 된 결과에 의존하기 전에 법적, 상업적 및 기술적 자문을 받아야합니다. 파이프 라인은 사용자가 모델에서 생성 한 결과에 의존하여 발생하는 어떠한 직접적인, 간접적인, 특수, 부수적 또는 결과적 손해에 대해서도 책임지지 않습니다.

## Support

도움이 필요하면 공식 [깃 허브 레포지토리](https://github.com/OptimalScale/LMFlow)에 이슈를 생성해주세요.

## Contributors
<a href="https://github.com/OptimalScale/LMFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OptimalScale/LMFlow" />
</a>

## Citation
이 repository를 유용하게 사용하셨다면 ⭐을 눌러주시고 다음을 통해 인용해주시면 감사하겠습니다.

```bibtex
@misc{lmflow,
  author = {Shizhe Diao and Rui Pan and Hanze Dong and KaShun Shum and Jipeng Zhang and Wei Xiong and Tong Zhang},
  title = {LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://optimalscale.github.io/LMFlow/}},
}
```
