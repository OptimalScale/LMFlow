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
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_ko.md">한국어</a> |
        <b>हिंदी</b>
    <p>
</h4>

यह चैटजीपीटी द्वारा अनुवादित हिंदी संस्करण है, यदि कोई त्रुटि हो, तो संबंधित योगदानकर्ताओं द्वारा संशोधित किया जा सकता है। इसके साथ ही यदि कोई सामग्री अंग्रेजी संस्करण से भिन्न हो या मेल नहीं खाती हो, तो कृपया अंग्रेजी संस्करण को ही मान्य रखें। धन्यवाद।

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)

एक विस्तारयोग्य, सुविधाजनक और दक्ष टूलबॉक्स जो बड़े मशीन लर्निंग मॉडल को finetune करने के लिए बनाया गया है, जो सभी समुदाय के उपयोगकर्ताओं के लिए उपलब्ध होने के साथ-साथ उपयोगकर्ता मित्रता, गति और विश्वसनीयता के साथ डिजाइन किया गया है।

लार्ज लैंग्वेज मॉडल फॉर ऑल। हमारी [दृष्टि](https://github.com/OptimalScale/LMFlow#vision) देखें।

<p align="center" width="100%">
<img src="../assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Latest News
* [2023-04-02] [Web service is online!](https://lmflow.com/)
* [2023-04-01] [Release Chinese checkpoints in model zoo: LLaMA-7B-tuned, LLaMA-13B-tuned, LLaMA-33B-tuned.](https://github.com/OptimalScale/LMFlow#model-zoo)
* [2023-04-01] [Release English checkpoints in model zoo: LLaMA-7B-medical, LLaMA-13B-medical, and LLaMA-33B-medical.](https://github.com/OptimalScale/LMFlow#model-zoo)
* [2023-03-27] [Support full tuning and lora tuning for all decoder models.](https://github.com/OptimalScale/LMFlow#supported-models) 
* [2023-03-27] [Tasked tuned model beats ChatGPT on medical domain](https://github.com/OptimalScale/LMFlow#model-performance)
* [2023-03-27] [Release code and checkpoints - version 0.0.1](https://optimalscale.github.io/LMFlow/)


## Demos

### वर्तमान में हमारी चेकपॉइंट डाउनलोड सेवा क्षमता पूर्ति पर है। हमने इसे समर्थन करने के लिए एक अतिरिक्त सर्वर आवंटित किया है। यदि आप "तूर्ता एचटीटीपी अनुरोधों की अधिक मात्रा" त्रुटि का सामना करते हैं, तो कृपया कुछ मिनट इंतजार करें और फिर से कोशिश करें। आपकी समझ और सहयोग के लिए धन्यवाद। 🙏

हम चार प्रकार के डेमो प्रदान करते हैं, जो निम्नलिखित हैं:

- ऑनलाइन सेवा: यदि आप कोई कोड चलाना नहीं चाहते और बस हमारे मॉडल का उपयोग करना चाहते हैं, तो हम आपके लिए अपने इंस्ट्रक्शन-ट्यून्ड LLaMA-7B और LLaMA-33B को डिप्लॉय करते हैं ताकि आप इन्हें आजमा सकें।
- कोलैब चैटबॉट (शैल): एक इंटरैक्टिव शैल-आधारित चैटबॉट जो आपको कोलैब पर एक चैटबॉट आसानी से डिप्लॉय करने की सुविधा प्रदान करता है।
- कोलैब चैटबॉट (वेब): एक इंटरैक्टिव वेब-आधारित चैटबॉट जो आपको कोलैब पर अपने खुद के चैटबॉट को आसानी से डिप्लॉय करने की सुविधा प्रदान करता है।
- स्थानीय डिप्लॉय: हम आपको अपने मॉडल / चैटबॉट को स्थानीय रूप से डिप्लॉय करने का एक तरीका भी प्रदान करते हैं, जिसका अर्थ है कि यदि आपके पास पर्याप्त संसाधन हैं, तो आप पिछले तीन विधियों से बहुत बड़े मॉडल को डिप्लॉय कर सकत


[![Code License](https://img.shields.io/badge/Online%20Service-Web-green.svg)](https://lmflow.com)
[![colab badge](https://img.shields.io/badge/Colab-(shell)%20%20chatbot:%20gpt--neo-orange?logo=google-colab&amp)](https://colab.research.google.com/drive/1gvW9S6peZY3qfljdBpBbCflqaII8quQW?usp=sharing)
[![colab badge](https://img.shields.io/badge/Colab-(web)%20%20chatbot:%20gpt--neo-blue?logo=google-colab&amp)](https://colab.research.google.com/drive/1LLtiiQO-ZIIFsTKxYzGWYX9BDRc-v8dq?usp=sharing)


### Online Service
>[वेब सेवा](https://lmflow.com/) पर आपका स्वागत है। हम देखभाल करते हुए LLaMA-7B-ट्यून्ड और LLaMA-33B-ट्यून्ड को ऑनलाइन पूर्वावलोकन के लिए डिप्लॉय करते हैं। उच्च वेबसाइट ट्रैफ़िक के कारण, कभी-कभी वेबसाइट प्रतिक्रिया नहीं दे पाती है। आप `सथानीय डिप्लॉय` पर जाकर भी चैटबॉट डिप्लॉय कर सकते हैं।

### Colab chatbot(shell)
<p align="center" width="100%">
<img src="../assets/colab-shell-chatbot-demo.png">
</p>

हम Google Colab के T4/P100/V100 GPU के साथ चैटबॉट का एक सरल शैली डेमो प्रदान करते हैं।
ध्यान दें कि प्रदान किए गए gpt-neo-2.7b मॉडल **एक बहुत ही कमजोर मॉडल** है, जो केवल अंग्रेजी का समर्थन करता है और कभी-कभी असंतोषजनक प्रतिक्रियाएं उत्पन्न कर सकता है। प्रदर्शन को बेहतर बनाने के लिए, उपयोगकर्ता अपने खुद के डेटासेट का उपयोग कर सकते हैं ताकि वह LMFlow के साथ एक बेहतर मॉडल प्राप्त कर सके। उपयोगकर्ता अन्य उपलब्ध डिकोडर-केवल मॉडलों का भी प्रयास कर सकते हैं, जो 🤗 [huggingface](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads),में उपलब्ध हैं।


```sh
./scripts/run_chatbot.sh {another-model-name}
```
### Colab chatbot(web)
हम Google Colab के T4/P100/V100 GPU के साथ चैटबॉट का एक सरल वेब डेमो प्रदान करते हैं। ध्यान दें कि प्रदान किए गए gpt-neo-2.7b मॉडल **एक बहुत ही कमजोर मॉडल** है, जो केवल अंग्रेजी का समर्थन करता है और कभी-कभी असंतोषजनक प्रतिक्रियाएं उत्पन्न कर सकता है।





### Local Deploy
यदि आप संसाधन रखते हैं और अपना खुद का मॉडल स्थानीय रूप से डिप्लॉय करना चाहते हैं, तो हम आपको एक आसान तरीका प्रदान करते हैं जिससे आप एक फ्लास्क सर्वर चला सकते हैं, जिससे आप एक बैकएंड लॉन्च कर सकते हैं (अन्य फ्रंटएंड को आगे सेवाएं प्रदान करने के लिए) और एक इंटरैक्टिव वेब फ्रंटएंड (आपको सीधे संचार करने की अनुमति देता है)।
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

LLaMA 33B (LoRA) की प्रदर्शन योग्यता एक एकल 8 * A100 सर्वर पर PubMedQA और MedMCQA के प्रशिक्षण स्प्लिट पर केवल **~16 घंटे** की फाइनट्यूनिंग से हासिल की जाती है।
अधिक प्रदर्शन, उदाहरण ट्यूनिंग परिणाम आदि के लिए कृपया हमारे [दस्तावेज़ीकरण](https://optimalscale.github.io/LMFlow/)
 का उल्लेख करें।
## Model Zoo
हमने प्रशिक्षित चेकपॉइंट को अधिक अभ्यास और अनुमान के लिए सभी के लिए ओपन-सोर्स कर दिया है।

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
🤗 huggingface में सभी [decoder models](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) का सहज तरीके से समर्थन किया गया है।
LLaMA, GPT2, GPT-Neo, Galactica, को पूरी तरह से टेस्ट किया गया है। हम जल्द ही एनकोडर मॉडल का समर्थन करेंगे।

## 1.Setup

हमारी पैकेजिंग लिनक्स ओएस (उबंटू 20.04) पर पूरी तरह से टेस्ट की गई है। अन्य ओएस प्लेटफॉर्म (MacOS, Windows) पूरी तरह से टेस्ट नहीं किए गए हैं।
आप कुछ अप्रत्याशित त्रुटियों से मिल सकते हैं. आप इसे पहले लिनक्स मशीन पर कोशिश कर सकते हैं या इसे अनुभव करने के लिए गूगल कोलेब इस्तेमाल कर सकते हैं.

```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```

## 2.Prepare Dataset
आप आसानी से उदाहरण ट्रेनिंग डेटासेट और टेस्ट डेटासेट डाउनलोड कर सकते हैं निम्नलिखित कमांड रन करके:
```bash
cd data
bash download.sh all
cd -
``` 

आप अपना खुद का डेटासेट भी निम्नलिखित स्वरूप में कनवर्ट करके उपयोग कर सकते हैं:
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

आप एक GPT-2 बेस मॉडल को फाइनट्यून करने के लिए `scripts/run_finetune.sh` को चला सकते हैं।
```sh
./scripts/run_finetune.sh
```

यदि आप अपनी मशीन सेटिंग को दर्शाने के लिए deepspeed के लिए तर्क प्रदान करना चाहते हैं, तो आप स्क्रिप्ट को उसके लिए संबंधित deepspeed तर्क पास कर सकते हैं। उदाहरण के लिए,

```sh
./scripts/run_finetune.sh "--num_gpus=8 --master_port 10001"
```

LoRA फाइनट्यूनिंग को सक्षम करने के लिए, आप इससे संबंधित जानकारी के लिए देख सकते हैं:
```sh
./scripts/run_finetune_with_lora.sh
```
जो एक ही तरीके से चलाया जा सकता है।

विस्तृत कॉन्फ़िगरेशन के लिए, आप इन स्क्रिप्ट को सीधे संशोधित कर सकते हैं। ये स्क्रिप्ट वास्तव में केवल पाइथन स्क्रिप्ट `examples/finetune.py` को बुलाते हैं, जिसे निम्न तरीके से चलाया जा सकता है:

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
यहां हम नंबर ऑफ एपोक को `--num_train_epochs` शून्य दर्ज करते हैं 0.01 ताकि फाइन-ट्यूनिंग प्रक्रिया त्वरित रूप से समाप्त हो सके। यदि आप एक बेहतर प्रदर्शन वाले मॉडल की इच्छा रखते हैं, तो वे हाइपरपैरामीटर को समायोजित करने में स्वतंत्र महसूस करें। आप निम्नलिखित कमांड का उपयोग कर सकते हैं।
```python
python examples/finetune.py -h
```
सभी संभव फाइन-ट्यूनिंग तर्क देखने के लिए आप निम्नलिखित कमांड का उपयोग कर सकते हैं। फाइन-ट्यून्ड मॉडल चेकपॉइंट `--output_dir` द्वारा निर्दिष्ट तर्क में सहेजा जाएगा, जो उपरोक्त उदाहरण में `output_models/finetune` है।
### 3.2 Run Evaluation

कोई व्यक्ति एक मौजूदा हगिंगफेस मॉडल के साथ सीधे मूल्यांकन चला सकता है, उदाहरण के लिए, GPT2 लार्ज चलाने के लिए, निम्नलिखित कमांड का उपयोग किया जा सकता है।
```sh
./scripts/run_evaluation.sh
```
या फिर संबंधित पायथन स्क्रिप्ट को चलाएं।
```python
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --dataset_path data/MedQA-USMLE/validation \
    --deepspeed examples/ds_config.json
```
फाइनट्यून्ड मॉडल लोड करने के लिए, सहेजे गए मॉडल चेकपॉइंट निर्दिष्ट डायरेक्टरी पथ के साथ `--model_name_or_path` निर्दिष्ट करें।

LoRA फाइनट्यून्ड मॉडल के लिए, निम्न दिए गए लिंक पर जा सकते हैं।

```sh
./scripts/run_evaluation_with_lora.sh
```

वे स्क्रिप्ट हमारी API के आधार पर बनाए गए उदाहरण `examples/*.py` को अधिकारित करते हैं। API से संबंधित अधिक उदाहरण के लिए, व्यक्तियों को `tests` में यूनिटटेस्ट में दिए गए विधियों का उल्लेख करना चाहिए।



## 4. Additional Notes
### 4.1 LLaMA Checkpoint

1. पहले, आपको [facebookresearch/llama](https://github.com/facebookresearch/llama) से LLaMA मॉडल का उपयोग प्राप्त करना होगा। आधिकारिक चेकपॉइंट डाउनलोड करें और उन्हें `${llama-path}` में सहेजें।

2. दूसरा, आधिकारिक चेकपॉइंट `${llama-path}` को चला कर HuggingFace समर्थित चेकपॉइंट `${llama-hf-path}` में रूपांतरित करें। कमांड लाइन में निम्नलिखित अंक चलाएँ:

    `python ./scripts/convert_llama_weights_to_hf.py --input_dir ${llama-path} --model_size 7B --output_dir ${llama-hf-path}/llama-7b-hf`

3. फिर आप `${llama-hf-path}/llama-7b-hf` को चेकपॉइंट पथ के रूप में सेट करके जाने के लिए तैयार हैं। मज़ा करें!

4. (वैकल्पिक) अब आपके पास मूल llama-7b-hf pretrained मॉडल है।
```sh
cd output_models && ./download.sh all && cd -
```
आप हमारे द्वारा finetuned मॉडल द्वारा मॉडल अंतर प्राप्त कर सकते हैं। `./scripts/run_evaluation_with_lora.sh` की तरह एक तरीके से,
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
अब आप finetuned llama मॉडल के साथ मूल्यांकन कर सकते हैं।

### 4.2 DeepSpeed Config
आप DeepSpeed को configs के तहत विन्यास कर सकते हैं। विवरण देखने के लिए [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/) Configuration का संदर्भ ले सकते हैं।

## 5. Model Release

### 5.1 Medical Model Checkpoints
आप हमारे चिकित्सा मॉडल चेकपॉइंट डाउनलोड करने के लिए निम्नलिखित स्क्रिप्ट चला सकते हैं:

```bash
cd output_models
bash download.sh medical_ckpt
cd -
```
आप Google Drive लिंक के माध्यम से भी हमारे मॉडल को सीधे डाउनलोड कर सकते हैं: [medical_ckpt.tar.gz](https://drive.google.com/file/d/1bnsQGNGNYchsOfiNyRAmL2fNiowbmFNw/view?usp=share_link)



### 5.2 Instruction Model Checkpoints
आप हमारे निर्देश मॉडल चेकपॉइंट डाउनलोड करने के लिए निम्नलिखित स्क्रिप्ट चला सकते हैं:
```bash
cd output_models
bash download.sh instruction_ckpt
cd -
```
आप Google Drive लिंक के माध्यम से भी हमारे मॉडल को सीधे डाउनलोड कर सकते हैं: [instruction_ckpt.tar.gz](https://drive.google.com/file/d/1d_ioQ-ViVweeifbsFSO4pczc3UORFHZO/view?usp=share_link)

### 5.3 Begin Reproduce

मॉडल चेकपॉइंट डाउनलोड करने के बाद, आप `LMFlow/scripts/run_evaluation_with_lora.sh` में `--lora_model_path` को `output_models/instruction_ckpt/llama7b-lora` (निर्देश के लिए उदाहरण के लिए llama-7b) से बदल सकते हैं और अपने कन्वर्ट किए गए ल्लामा मॉडल के साथ `--model_name_or_path` को बदल सकते हैं और इस शेल स्क्रिप्ट को चला सकते हैं ताकि परिणाम दोहराए जा सकें।

फिर आप हमारे [डॉक](https://optimalscale.github.io/LMFlow/) पर मॉडल के प्रदर्शन की जांच कर सकते हैं।

## Documentation
अधिक API संदर्भ और प्रायोगिक परिणामों के लिए कृपया हमारे [दस्तावेज़ीकरण](https://optimalscale.github.io/LMFlow/) का उल्लेख करें।
## Vision
नमस्ते! हम आपको बताने के लिए उत्साहित हैं कि हमारी कोड रिपॉजिटरी के आगामी रिलीज की घोषणा करने जा रहे हैं, जो एक पूर्ण LLM ट्रेनिंग प्रक्रिया को शामिल करती है, जो उपयोगकर्ताओं को अपने खुद के भाषा मॉडल तैयार करने और उन्हें प्रभावी ढंग से ट्रेन करने की अनुमति देती है।

हमारी कोड रिपॉजिटरी बस एक साधारण मॉडल नहीं है; इसमें पूर्ण ट्रेनिंग वर्कफ़्लो, मॉडल ऑप्टिमाइजेशन और टेस्टिंग टूल्स शामिल हैं। आप इसका उपयोग विभिन्न प्रकार के भाषा मॉडल तैयार करने के लिए कर सकते हैं, जिसमें बातचीत मॉडल, प्रश्न-उत्तर मॉडल और टेक्स्ट जनरेशन मॉडल आदि शामिल हैं।

इसके अतिरिक्त, हमारा लक्ष्य एक खुला और लोकतांत्रिक LLM साझा करने का मंच बनाना है जहां लोग अपने चेकपॉइंट और अनुभव साझा करके समुदाय के कौशलों को संगठित ढंग से सुधार सकते हैं। हम उन सभी लोगों का स्वागत करते हैं जो LLM में रूचि रखते हैं और एक खुले और मित्रवाही समुदाय का निर्माण करने में हमारे साथ शामिल होना चाहते हैं!

चाहे आप एक शुरुआती हों या एक विशेषज्ञ, हम यह मानते हैं कि आप इस मंच से लाभ उठा सकते हैं। आओ हम साथ मिलकर एक जीवंत और नवाचारी LLM समुदाय का निर्माण करें!
[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)

## Disclaimer

यह पैकेज बड़े मॉडल ट्यूनिंग के लिए एक संचालित और उपयोगकर्ता-मित्र पाइपलाइन प्रदान करने का उद्देश्य है। इसकी कार्यक्षमताएं उपयोगकर्ता द्वारा उपयोग के लिए संदर्भ के रूप में एक स्रोत के रूप में सेवा करती हैं। हालांकि, यह ध्यान देने योग्य है कि डेटा और पूर्व-प्रशिक्षित मॉडलों की तैयारी की जिम्मेदारी केवल उपयोगकर्ता की होती है। यह पैकेज उपयोगकर्ता द्वारा तैयार किए गए घटकों की सटीकता, पूर्णता, उपयोगिता या कानूनीता की गारंटी नहीं देता है। उपयोगकर्ताओं को मॉडल और डेटा की तैयारी से संबंधित सभी जोखिमों और दायित्वों को जानने और स्वीकार करने की आवश्यकता होगी, और इस पैकेज का उपयोग करने से पहले कानूनी, वाणिज्यिक और तकनीकी सलाह प्राप्त करनी होगी। उपयोगकर्ता द्वारा डेटा और पूर्व-प्रशिक्षित मॉडलों की तैयारी में अनुचितता से होने वाले किसी भी सीधे, अप्रत्यक्ष, विशेष, संय

हमारे चेकपॉइंट्स जो अंग्रेजी और चीनी दोनों संस्करणों में शामिल हैं, केवल शोध के उद्देश्यों के लिए प्रदान किए जाते हैं। इन चेकपॉइंट्स में शिक्षण डेटा ChatGPT भाषा मॉडल से उत्पन्न परिणामों को शामिल करता है। हम इन चेकपॉइंट्स के वितरण या उपयोग का अनुशंसित या प्रोत्साहित नहीं करते हैं वे केवल शोध के लिए प्रदान किए जाते हैं। इन चेकपॉइंट का उपयोग करने वाले उपयोगकर्ताओं की जिम्मेदारी होती है कि वे सुनिश्चित करें कि इन्हें सही और उचित ढंग से उपयोग किया जाता है।

यह भी महत्वपूर्ण है कि मॉडल द्वारा उत्पन्न परिणाम प्रायोजित मॉडल के प्रायोजन से संबंधित नहीं होते हैं और प्रायोजित मॉडल की सीधी जुड़ती नहीं हैं। इस पाइपलाइन द्वारा परिणामों की सटीकता, विश्वसनीयता, उपयोगीता और कानूनीता की गारंटी नहीं होती है। इसलिए, उपयोगकर्ताओं को परिणामों से जुड़े जोखिम और जिम्मेदारियों के बारे में भी जागरूक होना चाहिए और मॉडल द्वारा उत्पन्न परिणामों पर भरोसा करने से पहले कानूनी, वाणिज्यिक और तकनीकी सलाह लेनी चाहिए। इस पाइपलाइन को किसी भी प्रत्यक्ष, अप्रत्यक्ष, विशेष, आकस्मिक या परिणामी क्षति के लिए ज़िम्मेदार नहीं माना जाएगा, जो उपयोगकर्ता द्वारा मॉडल द्वारा उत्पन्न परिणामों पर आश्रित होते हैं।

## Support
यदि आपको किसी भी मदद की आवश्यकता हो तो, कृपया एक [Github](https://github.com/OptimalScale/LMFlow) इशु प्रस्तुत करें।
## Contributors
<a href="https://github.com/OptimalScale/LMFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OptimalScale/LMFlow" />
</a>

## Citation
यदि आपको यह रेपो उपयोगी लगता है, तो कृपया ⭐ देने और उद्धरण करने का विचार करें:

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
