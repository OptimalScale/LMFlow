<p align="center" width="100%">
<img src="assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README_es.md">English</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README_es.md">Español</a> |
        <b>日本語</b>
    <p>
</h4>

日文版はChatGPTによって翻訳されました。もし間違いがあれば、contributorに修正していただけると幸いです。また、英語版と内容に差異がある場合は、英語版を優先してください。

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/srGxyazbNs)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)

拡張性、利便性、効率性に優れた、大規模な機械学習モデルのファインチューニングに最適なツールボックスで、ユーザーフレンドリーで高速かつ信頼性があり、コミュニティ全体で利用可能な設計です。

すべての人のための大規模言語モデル。私たちの[ビジョン](https://github.com/OptimalScale/LMFlow#vision)をご覧ください

<p align="center" width="100%">
<img src="assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Latest News
* [2023-04-02] [Web service is online!](https://lmflow.com/)
* [2023-04-01] [Release Chinese checkpoints in model zoo: LLaMA-7B-tuned, LLaMA-13B-tuned, LLaMA-33B-tuned.](https://github.com/OptimalScale/LMFlow#model-zoo)
* [2023-04-01] [Release English checkpoints in model zoo: LLaMA-7B-medical, LLaMA-13B-medical, and LLaMA-33B-medical.](https://github.com/OptimalScale/LMFlow#model-zoo)
* [2023-03-27] [Support full tuning and lora tuning for all decoder models.](https://github.com/OptimalScale/LMFlow#supported-models) 
* [2023-03-27] [Tasked tuned model beats ChatGPT on medical domain](https://github.com/OptimalScale/LMFlow#model-performance)
* [2023-03-27] [Release code and checkpoints - version 0.0.1](https://optimalscale.github.io/LMFlow/)

## Demos

### 現在、私たちのチェックポイントダウンロードサービスはキャパシティに達しています。1つのサーバーを割り当ててサポートしています。もし「too many HTTP requests」のエラーが表示された場合は、数分待ってから再度お試しください。ご理解いただきありがとうございます。🙏

私たちは以下の4種類のデモを提供しています。
- オンラインサービス：コードを実行する必要がなく、私たちのモデルを試したいだけの場合、説明にチューニングされたLLaMA-7BとLLaMA-33Bをデプロイしています。
- Colabチャットボット（シェル）：対話型のシェルベースのチャットボットで、簡単にColab上でチャットボットをデプロイできます。
- Colabチャットボット（Web）：対話型のWebベースのチャットボットで、簡単に自分自身のチャットボットをColab上でデプロイできます。
- ローカルデプロイ：自分のモデル/チャットボットをローカルにデプロイする方法も提供しています。つまり、リソースが十分であれば、前述の3つの方法よりもはるかに大きなモデルをデプロイできます。


[![Code License](https://img.shields.io/badge/Online%20Service-Web-green.svg)](https://lmflow.com)
[![colab badge](https://img.shields.io/badge/Colab-(shell)%20%20chatbot:%20gpt--neo-orange?logo=google-colab&amp)](https://colab.research.google.com/drive/1gvW9S6peZY3qfljdBpBbCflqaII8quQW?usp=sharing)
[![colab badge](https://img.shields.io/badge/Colab-(web)%20%20chatbot:%20gpt--neo-blue?logo=google-colab&amp)](https://colab.research.google.com/drive/1LLtiiQO-ZIIFsTKxYzGWYX9BDRc-v8dq?usp=sharing)


### Online Service
> [Webサービス](https://lmflow.com/)にアクセスしていただきありがとうございます。LLaMA-7B-tunedとLLaMA-33B-tunedをオンラインでプレビュー用にデプロイしています。ウェブサイトのトラフィックが高いため、サイトが応答しないことがあります。その場合は、「ローカルデプロイ」を参照してチャットボットをデプロイすることもできます。.

### Colab chatbot(shell)
<p align="center" width="100%">
<img src="assets/colab-shell-chatbot-demo.png">
</p>

私たちは、Google ColabのT4/P100/V100 GPUを使用した、シンプルなシェルデモのチャットボットを提供しています。
提供されるgpt-neo-2.7bモデルは、**かなり弱いモデル**であり、英語のみをサポートしており、時には不十分な応答を生成することがあります。パフォーマンスを改善するには、ユーザー自身のデータセットを使用してファインチューニングを行い、LMFlowでより良いモデルを取得することができます。また、他の利用可能なデコーダー専用モデルも試すことができます。🤗 [huggingface](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)で提供されています。

```sh
./scripts/run_chatbot.sh {another-model-name}
```
### Colab chatbot(web)
私たちは、Google ColabのT4/P100/V100 GPUを使用した、シンプルなウェブデモのチャットボットを提供しています。
提供されるgpt-neo-2.7bモデルは、**かなり弱いモデル**であり、英語のみをサポートしており、時には不十分な応答を生成することがあります。


### Local Deploy
もしリソースを持っていてローカルに独自のモデルをデプロイしたい場合は、以下の手順で簡単にFlaskサーバーを実行してバックエンドを起動し、対話型のWebフロントエンドを起動することができます。
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

LLaMA 33B（LoRA）のパフォーマンスは、単一の8 \ * A100サーバーでPubMedQAおよびMedMCQAのトレーニングスプリットで**約16時間**のファインチューニングで達成されます。
Instruction tuningの結果を含む、より詳細なパフォーマンスについては、当社の[ドキュメンテーション](https://optimalscale.github.io/LMFlow/)を参照してください。

## Model Zoo
当社はトレーニング済みのチェックポイントをオープンソース化し、誰でも追加のトレーニングや推論に使用できるようにしました。

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

🤗 Hugging Faceのすべての[decoder models](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)を完全にサポートし、LLaMA、GPT2、GPT-Neo、Galacticaを完全にテストしました。エンコーダーモデルも近日中にサポートする予定です。



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
以下のコマンドを実行することで、簡単にトレーニング用のサンプルデータセットとテスト用データセットをダウンロードできます。
```bash
cd data
bash download.sh all
cd -
``` 

独自のデータセットを使用する場合は、以下の形式に変換するだけで使用できます。

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

以下のように、GPT-2ベースモデルの微調整を実行するには、`scripts/run_finetune.sh` を実行できます。

```sh
./scripts/run_finetune.sh
```

もし、あなたのマシンの設定を反映するためにdeepspeedに引数を提供したい場合は、対応するdeepspeedの引数をスクリプトに渡すことができます。例えば、以下のようになります。
```sh
./scripts/run_finetune.sh "--num_gpus=8 --master_port 10001"
```

LoRAの微調整を有効にするには、以下を参照してください。
```sh
./scripts/run_finetune_with_lora.sh
```
同様の方法で実行できます。

詳細な設定については、これらのスクリプトを直接変更することができます。これらのスクリプトは実際には、Pythonスクリプト`examples/finetune.py`を呼び出しているだけです。以下のように実行することができます。



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
ここでは、エポック数`--num_train_epochs`を`0.01`に設定して、微調整プロセスを迅速に終了できるようにしています。より良い性能を持つモデルを取得したい場合は、これらのハイパーパラメータを自由に調整してください。すべての可能な微調整引数を表示するには、
```python
python examples/finetune.py -h
```
を実行できます。微調整されたモデルのチェックポイントは、`--output_dir`で指定された引数に保存されます。上記の例では、`output_models/finetune`に保存されます。
### 3.2 Run Evaluation

既存のHugging Faceモデルで直接評価を実行することができます。たとえば、GPT2 largeを実行するには、以下のように実行できます。
```sh
./scripts/run_evaluation.sh
```
または、対応するPythonスクリプトを実行することもできます
```python
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --dataset_path data/MedQA-USMLE/validation \
    --deepspeed examples/ds_config.json
```
微調整済みモデルをロードするには、保存されたモデルのチェックポイントディレクトリパスを`--model_name_or_path`で指定します。

LoRAで微調整されたモデルについては、以下を参照してください
```sh
./scripts/run_evaluation_with_lora.sh
```

これらのスクリプトは、当社のAPIを基に構築された例`examples/*.py`を呼び出します。APIに関連するより詳細な例については、`tests`のユニットテスト内のメソッドを参照してください。

## 4. Additional Notes
### 4.1 LLaMA Checkpoint

1. まず、[facebookresearch/llama](https://github.com/facebookresearch/llama)からLLaMAモデルへのアクセスを取得する必要があります。公式のチェックポイントをダウンロードし`${llama-path}`に保存します。
2. 次に、公式のチェックポイント`${llama-path}`をHuggingFaceがサポートするチェックポイント`${llama-hf-path}`に変換するには、次を実行してください。
    `python ./scripts/convert_llama_weights_to_hf.py --input_dir ${llama-path} --model_size 7B --output_dir ${llama-hf-path}/llama-7b-hf`

3. これで、`${llama-hf-path}/llama-7b-hf`へのチェックポイントパスを設定することで、準備ができました。お楽しみください！

4. （オプション）これで、元のllama-7b-hf事前学習モデルがあります。を
```sh
cd output_models && ./download.sh all && cd -
```
使用して、当社によって微調整されたモデルの差分を取得できます。`./scripts/run_evaluation_with_lora.sh`と同様の方法で、以下を実行してください。
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
これで、微調整されたllamaモデルで評価を実行できます。

### 4.2 DeepSpeed Config
configsフォルダー内でdeepspeedを設定できます。詳細については、[DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)を参照してください。
## 5. Model Release

### 5.1 Medical Model Checkpoints
当社の医療モデルのチェックポイントをダウンロードするには、以下のスクリプトを実行できます:

```bash
cd output_models
bash download.sh medical_ckpt
cd -
```
また、以下のGoogleドライブリンクから直接当社のモデルをダウンロードすることもできます : [medical_ckpt.tar.gz](https://drive.google.com/file/d/1bnsQGNGNYchsOfiNyRAmL2fNiowbmFNw/view?usp=share_link)

### 5.2 Instruction Model Checkpoints
同様に、以下のスクリプトを実行して、当社の説明書モデルのチェックポイントをダウンロードできます:
```bash
cd output_models
bash download.sh instruction_ckpt
cd -
```

また、以下のGoogleドライブリンクから直接当社のモデルをダウンロードすることもできます:[instruction_ckpt.tar.gz](https://drive.google.com/file/d/1d_ioQ-ViVweeifbsFSO4pczc3UORFHZO/view?usp=share_link)

### 5.3 Begin Reproduce

モデルのチェックポイントをダウンロードした後、`LMFlow/scripts/run_evaluation_with_lora.sh`内の`--lora_model_path`を`output_models/instruction_ckpt/llama7b-lora(instruction`用の例)に置き換え、`--model_name_or_path`を変換済みのllamaモデルに置き換えます。そして、このシェルスクリプトを実行して、結果を再現できます。

その後、[Doc](https://optimalscale.github.io/LMFlow/).でモデルのパフォーマンスを確認できます。

## Documentation
より詳しいAPIリファレンスや実験結果については、[ドキュメント](https://optimalscale.github.io/LMFlow/)を参照してください。

## Vision
こんにちは！ 私たちは、完全なLLMトレーニングプロセスを含むコードリポジトリの近日リリースをお知らせできることを喜んでいます。これにより、ユーザーは自分自身の言語モデルを迅速に構築し、効果的にトレーニングすることができます。

私たちのコードリポジトリは単なるモデルだけでなく、完全なトレーニングワークフロー、モデルの最適化、およびテストツールを含んでいます。会話モデル、質問応答モデル、テキスト生成モデルなど、さまざまな種類の言語モデルを構築するために使用できます。

さらに、私たちは、人々がチェックポイントや経験を共有し、コミュニティのスキルを集団で向上させることができるオープンで民主的なLLM共有プラットフォームを作成することを目指しています。LLMに興味のある人は誰でも参加し、オープンでフレンドリーなコミュニティの構築に参加することを歓迎します！

初心者でもエキスパートでも、私たちはこのプラットフォームから利益を得ることができると信じています。一緒に活気ある革新的なLLMコミュニティを築いていきましょう！

[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/srGxyazbNs)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)

## Disclaimer
このパッケージは、大規模モデルの調整のための簡素化されたユーザーフレンドリーなパイプラインを提供することを目的としています。その機能はユーザーによって参照されることを意図しており、データと事前学習済みモデルの準備に関する責任はユーザーに完全にあります。このパッケージは、ユーザーの準備からのコンポーネントの正確性、完全性、適用性、および合法性を保証しません。ユーザーはモデルとデータの準備に関連するすべてのリスクと責任を認識し、本パッケージを利用する前に法的、商業的、および技術的なアドバイスを受ける必要があります。パイプラインは、ユーザーのデータと事前学習済みモデルの不適切な準備によって生じた直接的、間接的、特別な、付随的、または結果的な損害について責任を負いません。

私たちのチェックポイントには、英語版と中国語版の両方が含まれており、研究目的にのみ提供されています。これらのチェックポイントに含まれるトレーニングデータには、ChatGPT言語モデルから生成された結果が含まれています。これらのチェックポイントの配布や使用を商業目的で推奨または促進することはできません。これらのチェックポイントのユーザーは、正しく適切に使用されるように責任を負う必要があります。

モデルによって生成された結果は確率モデルに基づいており、このパイプラインと直接関係があるわけではありません。結果の正確性、信頼性、適用性、法的性質は、このパイプラインによって保証されるものではありません。したがって、ユーザーは結果に関連するリスクと責任を認識し、法的、商業的、技術的なアドバイスを受けてから、モデル生成の結果に依存する必要があります。このパイプラインは、ユーザーがモデル生成の結果に依存することによって生じる直接的、間接的、特別、偶発的、または結果的な損害について、一切責任を負いません。

## Support
何かお困りのことがございましたら、[Github](https://github.com/OptimalScale/LMFlow)のissueにご投稿ください。

## Contributors
<a href="https://github.com/OptimalScale/LMFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OptimalScale/LMFlow" />
</a>

## Citation
もしこのリポジトリが役立った場合は、ぜひ⭐をつけて引用してください。

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
