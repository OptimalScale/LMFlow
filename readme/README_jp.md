<p align="center" width="100%">
<img src="../assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README.md">English</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_es.md">Español</a> |
        <b>日本語</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_ko.md">한국어</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_hindi.md">हिंदी</a>
    <p>
</h4>

> [!NOTE]
> This README file was translated by LLM for reference only. Japanese speakers are welcome to submit PRs to polish the document!  

> [!NOTE]  
日本語版はChatGPTによって翻訳されました。もし間違いがあれば、contributorに修正していただけると幸いです。また、英語版と内容に差異がある場合は、英語版を優先してください。

[![Website](https://img.shields.io/badge/Website-Demo-20B2AA.svg)](https://lmflow.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/Discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://ibb.co/ZhM4hhn)

拡張性、利便性、効率性に優れた、大規模な機械学習モデルのファインチューニングに最適なツールボックスで、ユーザーフレンドリーで高速かつ信頼性があり、コミュニティ全体で利用可能な設計です。


<p align="center" width="100%">
<img src="../assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Latest News
* [2024-04-25] :rocket: 会話テンプレートのサポート！最新の[Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B)と[Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)の会話テンプレートを事前設定しました。また、`chatml`などのよく使用されるテンプレートも用意しています（すべてのテンプレートは[こちら](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#conversation-template)を参照してください）。さらに、追加の事前設定済みテンプレートを追加しています。シェルスクリプトに対応する`--conversation_template`を追加するだけで、準備完了です！ :rocket:  
* [2024-03-27] [LISA](https://arxiv.org/abs/2403.17919) に対応 —— オフロード不要、24GのGPUで7Bモデルをトレーニング！  
* [2023-09-11] [スペキュラティブ・デコーディング](https://arxiv.org/abs/2211.17192) をサポート、使用方法や簡単な性能統計については [使用ガイド](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) を参照してください。
* [2023-08-14] [位置補間（Linear & NTK scaling）](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md) を使用したLLaMAのコンテキストウィンドウを拡張する機能をサポートしています。
* [2023-08-07] [Flash Attention-2](https://crfm.stanford.edu/2023/07/17/flash2.html) をサポートしています。詳細は[Flash Attentionの使用ガイド](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)を参照してください。


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
私たちのリポジトリはすでにLinux（Ubuntu 20.04）で包括的なテストを完了しています。他のオペレーティングシステムプラットフォーム（MacOS、Windows）は完全にテストされていませんので、予期しないエラーが発生する可能性があります。まずLinux/Windows WSLで試してみるか、またはGoogle Colabをご利用ください。
CUDA 10.3-11.7については、`v0.0.5`またはそれ以前のバージョンを使用することをお勧めします。11.7よりも新しいCUDAの場合は、より良い体験を得るために、安定したブランチ`>= v0.0.6`を使用してください。
```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```

### Prepare Dataset
当社の[公式ドキュメント（英語版）](https://optimalscale.github.io/LMFlow/examples/DATASETS.html)を参照してください。公式ドキュメントは現在翻訳中ですので、しばらくお待ちください。

### Fine-Tuning (Full)
全パラメーターファインチューニングは、モデルのすべてのパラメーターを更新します。GPT-2の全パラメーターファインチューニングの例を以下に示します：

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

> [!TIP]
> 対話データセットに対話テンプレートを指定するには、`--conversation_template`パラメータを追加します。
> 
> <details><summary>Llama-3-8Bに対話データセットテンプレートを指定する例</summary>  
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
[LISA](https://arxiv.org/abs/2403.17919) は、**メモリ効率** の高いファインチューニングアルゴリズムであり、メモリとランダムに解凍された層の間でのバランスを取ることができます。以下のスクリプトは現在、**単一のGPU** 上でのみテストされています。最新情報にご注意ください！ :smile:
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
> <details><summary>例: Llama-2-7Bの対話データセットテンプレートの指定</summary>  
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

### Fine-Tuning (LoRA)
LoRAは、全パラメータ微調整よりも効率的なパラメータ効率微調整アルゴリズムです。
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path facebook/galactica-1.3b \
  --dataset_path data/alpaca/train_conversation \
  --output_lora_path output_models/finetuned_galactica_lora
```

> [!TIP]
> <details><summary>例：Llama-2-7Bに対する対話データセットのテンプレートを指定する</summary>  
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
></details>
>
><details><summary>LoRA重みの結合</summary>
>
>以下のコマンドを使用して、LoRAの重みと元のモデルを結合できます:  
>```sh
>./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>

### Inference
微調が終了したら、以下のコマンドを使用してモデルと対話できます。
```sh
./scripts/run_chatbot.sh output_models/finetuned_gpt2
```

### Deployment
ローカルでモデルを展開したい場合、GradioをベースにしたチャットボットUIが提供されています。
以下のコマンドでrobin-7bのデモを起動できます。詳細は次のとおりです：
```sh
pip install gradio
python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path YOUR-LLAMA  --lora_model_path ./robin-7b --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:"       --end_string "#" --max_new_tokens 200
```

### Evaluation
[LMFlow Benchmark](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418) はオープンソースLLMの自動評価フレームワークです。我々はNegative Log Likelihood (NLL) を使用して、LLMのチャット、一般的な推論、および命令に従う能力など、さまざまな側面を評価します。お手持ちのモデルを評価するために、LMFlow Benchmarkをご利用ください。そして、[モデルの比較](https://docs.google.com/spreadsheets/d/1JYh4_pxNzmNA9I0YM2epgRA7VXBIeIGS64gPJBg5NHA/edit?usp=sharing)にご参加ください。

GPT-2 XLを例に挙げますと、次のコマンドを使用して評価を開始します：
```sh
./scripts/run_benchmark.sh --model_name_or_path gpt2-xl
```
`--model_name_or_path`は必須のパラメータであり、Hugging Faceのモデル名またはモデルのローカルパスを渡すことができます。
評価結果は、`./output_dir/gpt2-xl_lmflow_chat_nll_eval`、`./output_dir/gpt2-xl_all_nll_eval`、および `./output_dir/gpt2-xl_commonsense_qa_eval`の`benchmark.log`で確認できます。


## Supported Features
<details> <summary>微調加速＆メモリ最適化</summary>

* LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning

  LISAはメモリ効率の高いLLMファインチューニングアルゴリズムです。微調整プロセス中に層を選択的に凍結することにより、LISAは既存のファインチューニング方法（LoRAなど）を超えています。詳細については[論文](https://arxiv.org/abs/2403.17919)をご覧ください。
  LISAを使用するには、トレーニングコマンドでパラメータ `--use_lisa 1` を指定します。アクティブ化される層の数を `--lisa_activated_layers 2` で制御し、フリーズされる層の間隔を `--lisa_step_interval 20` で調整できます。

* LoRA

  LoRAは、全パラメータ微調整よりも効率的なパラメータ効率（parameter-efficient）の微調整アルゴリズムです。詳細はこちらを参照してください：[微調（LoRA）](#fine-tuning-lora)。

* FlashAttention

  FlashAttention-1とFlashAttention-2をサポートしています。詳細については[FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)をご覧ください。

* Gradient Checkpointing

  [Gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)は、メモリ最適化技術の一種であり、計算をメモリとの交換により显存の使用量を削減します。トレーニングコマンドに `--gradient_checkpointing` を追加すると使用できます。

* Deepspeed Zero3

  LMFlowは[Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)をサポートしています。我々は使いやすい [deepspeed設定ファイル](https://github.com/OptimalScale/LMFlow/blob/main/configs/ds_config_zero3.json) を提供しています。

</details>


<details> <summary>推論の高速化</summary>

* LLaMA CPU推論
  
  [llama.cpp](https://github.com/ggerganov/llama.cpp)に感謝します。これにより、誰もがCPU上で自分のLLaMA（4ビット量子化）を実行できるようになりました！LLaMA LoRA重みを`.pt`ファイルに変換するスクリプトを提供しており、`convert-pth-to-ggml.py`を使用してモデルを量子化するだけで、LLaMA CPU推論を行うことができます。

* FlashAttention
  
  FlashAttention-1とFlashAttention-2をサポートしています。詳細はこちらをご覧ください：[FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md)。

</details>


<details> <summary>長文</summary>

* LLaMAモデルの位置補間（Position Interpolation）

  位置補間（Linear & NTK scaling）を使用してLLaMAのコンテキストウィンドウを拡張することができます。詳細はこちら：[位置補間](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md)。

</details>


<details> <summary>モデルのカスタマイズ</summary>

* 語彙の拡張

  独自のsentencepiece tokenizerをトレーニングし、それをモデルに含まれるhuggingface tokenizerとマージします。詳細はこちら：[語彙の拡張](https://github.com/OptimalScale/LMFlow/blob/main/scripts/vocab_extension)。

</details>


<details> <summary>マルチモーダル</summary>

* マルチモーダルチャットボット

  LMFlowはマルチモーダル（画像、テキスト）入力をサポートしています。詳細はこちら：[LMFlowマルチモーダルチャットボット](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_vis_chatbot_gradio_minigpt4.sh)。

</details>


## Support
何かお困りのことがございましたら、[GitHub](https://github.com/OptimalScale/LMFlow)のissueにご投稿ください。


## License
このプロジェクトに含まれるコードはApache 2.0ライセンスで提供されています。このプロジェクトに含まれるモデルを商業目的で使用したい場合は、プロジェクトの開発者に連絡して許可を取得してください。


## Citation
もしこのリポジトリが役立った場合は、ぜひ⭐をつけて引用してください。

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