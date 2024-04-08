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
* [2024-03-27] :rocket: 支持 [LISA](https://arxiv.org/abs/2403.17919) —— 无需offloading，在24G显存的GPU上训练7B模型！ :rocket:
* [2023-09-11] 支持 [投机解码(speculative decoding)](https://arxiv.org/abs/2211.17192)， 点击 [使用指南](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) 查看使用方法和简单的性能统计。
* [2023-08-14] 支持通过位置插值（Postion Interpolation）（Linear & NTK scaling）扩展LLaMA的上下文窗口，查看详情：[位置插值](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md)。
* [2023-08-07] 支持 [Flash Attention-2](https://crfm.stanford.edu/2023/07/17/flash2.html)，查看详情：[Flash Attention使用指南](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)。
* [2023-08-02] 支持 [Llama2](https://ai.meta.com/llama/)，[ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b)，[Baichuan](https://huggingface.co/baichuan-inc/Baichuan-7B)。


## Documentation
请参考我们的[Documentation](https://optimalscale.github.io/LMFlow/)获取更多API参考和实验结果信息。

## Vision
我们很高兴地开源LMFlow代码库，其中包括了完整的大模型训练流程，能够快速、高效地训练和部署自己的语言模型。

我们的代码库不仅仅是一个简单的模型； 它包括完整的训练流程、模型权重和测试工具。 您可以使用它来构建各种类型的语言模型，包括对话模型、问答模型和文本生成模型等。

此外，我们旨在创建一个开放和民主的大模型共享平台，任何人都可以在这个平台上分享训练模型权重和经验。 我们欢迎任何对大模型感兴趣的人参与进来，与我们一起建设一个开放友好的社区！

无论您是初学者还是专家，我们相信大家都能从这个平台中获益。让我们共同努力，建立一个充满活力和创新的大模型社区！

[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![WeChat badge](https://img.shields.io/badge/微信-加入-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/05/i8gG4z.jpeg)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)

## Disclaimer

此软件包旨在为大型模型调整提供简化和用户友好的流程。其功能可作为用户参考并供用户使用。然而，需要注意的是，数据和预训练模型的准备工作完全由用户负责。本软件包不保证用户准备组件的准确性、完整性、适用性或合法性。用户必须了解并承担与模型和数据准备相关的所有风险和责任，并在使用本软件包之前获取法律、商业和技术建议。该流程不应对用户不当准备数据和预训练模型所导致的任何直接、间接、特殊、偶然或后果性损害负责。

我们提供的检查点仅供研究目的使用，包括英文和中文版本。这些检查点包含ChatGPT语言模型生成的结果。我们不支持或鼓励将这些检查点用于商业目的的分发或使用。这些检查点的用户应当负责确保正确和适当地使用它们。

还需要强调的是，模型生成的结果是基于概率模型，与此流程没有直接关系。本流程不保证结果的准确性、可靠性、适用性和合法性。因此，在依赖模型生成的结果之前，用户还必须了解与结果相关的风险和责任，并寻求法律、商业和技术建议。该流程不应对用户依赖模型生成的结果所导致的任何直接、间接、特殊、偶然或后果性损害负责。

## Support
如果您需要任何帮助，请提交[Github](https://github.com/OptimalScale/LMFlow)问题。

## 协议
本项目所含代码采用Apache 2.0协议。如果您希望将本项目所含模型用于商业用途，请填写并签署[本文件](https://docs.google.com/forms/d/e/1FAIpQLSertnFbm2_aELsPMwOu_DhAu3p7bQgv8_MWSug7D80AyzPLhg/viewform?usp=pp_url)取得授权。

## Contributors
<a href="https://github.com/OptimalScale/LMFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OptimalScale/LMFlow" />
</a>

## Citation
如果您觉得我们的软件包有用，欢迎点赞⭐、fork、转发和引用。谢谢大家的支持！

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
