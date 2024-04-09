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

>Disclaimer: 
The Korean README file was translated by LLM for reference only. Korean speakers are welcome to submit a PR to polish the document!
>면책조항: 
한국어 README 파일은 참고용으로 LLM에 의해 번역되었습니다. 한국어 사용자들은 문서를 개선하기 위해 PR을 제출할 것을 환영합니다!

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

