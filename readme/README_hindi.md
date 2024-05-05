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

> [!NOTE]
> This README file was translated by LLM for reference only. Hindi speakers are welcome to submit PRs to polish the document!  

> [!NOTE]
यह चैटजीपीटी द्वारा अनुवादित हिंदी संस्करण है, यदि कोई त्रुटि हो, तो संबंधित योगदानकर्ताओं द्वारा संशोधित किया जा सकता है। इसके साथ ही यदि कोई सामग्री अंग्रेजी संस्करण से भिन्न हो या मेल नहीं खाती हो, तो कृपया अंग्रेजी संस्करण को ही मान्य रखें। धन्यवाद।

[![Website](https://img.shields.io/badge/Website-Demo-20B2AA.svg)](https://lmflow.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/Discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://ibb.co/ZhM4hhn)

एक विस्तारयोग्य, सुविधाजनक और दक्ष टूलबॉक्स जो बड़े मशीन लर्निंग मॉडल को finetune करने के लिए बनाया गया है, जो सभी समुदाय के उपयोगकर्ताओं के लिए उपलब्ध होने के साथ-साथ उपयोगकर्ता मित्रता, गति और विश्वसनीयता के साथ डिजाइन किया गया है।

<p align="center" width="100%">
<img src="../assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Latest News
* [2024-04-25] :rocket: बातचीत टेम्पलेट का समर्थन! हमने नवीनतम [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B) और [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) बातचीत टेम्पलेट को पूर्वनिर्धारित किया है, साथ ही कुछ अक्सर प्रयुक्त टेम्पलेट जैसे `chatml` भी (सभी टेम्पलेट यहाँ देखें [यहाँ](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#conversation-template)), और हम अधिक पूर्वनिर्धारित टेम्पलेट जोड़ने पर काम कर रहे हैं। शैल अनुक्रम में संबंधित `--conversation_template` को शैल अनुक्रम में जोड़ें और आप तैयार हैं! :rocket:  
* [2024-03-27] [LISA](https://arxiv.org/abs/2403.17919) का समर्थन —— 24जीबी जीपीयू पर 7B मॉडल का प्रशिक्षण बिना ऑफलोडिंग के!  
* [2023-09-11] [स्पेक्युलेटिव डिकोडिंग](https://arxiv.org/abs/2211.17192) का समर्थन, इस्तेमाल के तरीके और साधारण प्रदर्शन आँकड़े देखने के लिए [उपयोग गाइड](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) पर क्लिक करें।
* [2023-08-14] [पोजीशन इंटरपोलेशन](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md) के माध्यम से LLaMA की संदर्भ विंडो को विस्तारित करने का समर्थन (लीनियर और NTK स्केलिंग)।
* [2023-08-07] [फ्लैश एटेंशन-2](https://crfm.stanford.edu/2023/07/17/flash2.html) का समर्थन, अधिक जानकारी के लिए [फ्लैश एटेंशन उपयोग गाइड](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) देखें।


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
हमारे रेपो को Linux (Ubuntu 20.04) पर परीक्षण किया गया है। अन्य ऑपरेटिंग सिस्टम प्लेटफॉर्म (MacOS, Windows) को पूरी तरह से परीक्षण नहीं किया गया है, इसलिए कुछ अपेक्षित त्रुटियों का सामना कर सकता है। Linux/Windows WSL पर प्रयोग करने या Google Colab का उपयोग करके अनुभव करने की सिफारिश की जाती है।

CUDA 10.3-11.7 के लिए, `v0.0.5` या इससे पुराने संस्करणों का उपयोग करने की सिफारिश की जाती है। 11.7 से अधिक CUDA के लिए, बेहतर अनुभव के लिए हमारी स्थिर शाखा `>= v0.0.6` का उपयोग करें।
```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```

### Prepare Dataset
आप हमारी [आधिकारिक दस्तावेज़ीकरण (अंग्रेजी में)](https://optimalscale.github.io/LMFlow/examples/DATASETS.html) को देखें। आधिकारिक दस्तावेज़ीकरण अनुवाद के प्रक्रिया में है, कृपया धैर्य रखें।

### Fine-Tuning (Full)
मॉडल को पूर्ण पैरामीटर फ़ाइन ट्यूनिंग करने से सभी पैरामीटर अपडेट होते हैं। GPT-2 का एक पूर्ण पैरामीटर फ़ाइन ट्यूनिंग का उदाहरण निम्नलिखित है:

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

>[!TIP]
>आप बातचीत डेटासेट के लिए बातचीत टेम्पलेट को निर्दिष्ट करने के लिए `--conversation_template` पैरामीटर को जोड़कर कर सकते हैं।
>
><details><summary>उदाहरण: Llama-3-8B के लिए बातचीत डेटासेट टेम्पलेट का निर्दिष्ट करें</summary>
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
></details>

### Fine-Tuning (LISA)
[LISA](https://arxiv.org/abs/2403.17919) एक **मेमरी-एफिशिएंट (memory-efficient)** फ़ाइन ट्यूनिंग एल्गोरिदम है, जो मेमरी और रैंडम अनफ्रोज़न लेयरों के बीच संतुलन स्थापित करता है। निम्नलिखित स्क्रिप्ट अब **एकल GPU** पर ही टेस्ट किया गया है। हमारे नवीनतम अपडेट पर ध्यान दें! :smile:
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
> <details><summary>उदाहरण: Llama-2-7B के लिए बातचीत डेटा सेट टेम्पलेट का निर्दिष्ट करें</summary>  
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
LoRA एक पैरामीटर-सुसंगत (parameter-efficient) फाइन-ट्यूनिंग एल्गोरिथ्म है जो पूर्ण-पैरामीटर फाइन-ट्यूनिंग से अधिक दक्ष है।
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path facebook/galactica-1.3b \
  --dataset_path data/alpaca/train_conversation \
  --output_lora_path output_models/finetuned_galactica_lora
```

> [!TIP]
> <details><summary>उदाहरण: Llama-2-7B के लिए बातचीत डेटा सेट टेम्पलेट निर्दिष्ट करें</summary>  
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
> <details><summary>LoRA वज़न को मिलाना</summary>
>
>निम्नलिखित आदेश का उपयोग करके LoRA वज़न और मूल मॉडल को मिलाया जा सकता है:  
>```sh
>./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>

### Inference
एक बार फ़ाइन-ट्यूनिंग समाप्त हो जाने पर, आप निम्न आदेशों का उपयोग करके मॉडल के साथ इंटरैक्ट कर सकते हैं।
```sh
./scripts/run_chatbot.sh output_models/finetuned_gpt2
```

### Deployment
यदि आप अपने मॉडल को स्थानीय रूप से डिप्लॉय करना चाहते हैं, तो हम ग्राडियो पर आधारित चैट रोबोट UI प्रदान करते हैं।
निम्नलिखित कमांड robin-7b के डेमो को शुरू कर सकते हैं, कृपया संदर्भ के लिए:
```sh
pip install gradio
python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path YOUR-LLAMA  --lora_model_path ./robin-7b --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:"       --end_string "#" --max_new_tokens 200
```

### Evaluation
मुद्रित (छापे) [LMFlow Benchmark](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418) एक स्वत: एकांत मूल्यांकन के लिए एक फ्रेमवर्क है जो ओपन सोर्स एलएलएम के लिए बनाया गया है। हम विभिन्न पहलुओं का मूल्यांकन करने के लिए नेगेटिव लॉग लाइकलीहुड (एनएलएल) का उपयोग करते हैं, जैसे: चिटचट, सामान्य बुद्धिमत्ता और निर्देशों का पालन। आप अपने पास के मॉडल को मूल्यांकन करने के लिए LMFlow Benchmark का उपयोग करने का स्वागत करते हैं, और हमारे [मॉडल तुलना (LLM comparision)](https://docs.google.com/spreadsheets/d/1JYh4_pxNzmNA9I0YM2epgRA7VXBIeIGS64gPJBg5NHA/edit?usp=sharing) में शामिल होने के लिए।

GPT-2 XL को उदाहरण के रूप में, निम्नलिखित आदेश का पालन करके मूल्यांकन शुरू करें:
```sh
./scripts/run_benchmark.sh --model_name_or_path gpt2-xl
```
`--model_name_or_path` एक आवश्यक पैरामीटर है, जिसे हगिंगफेस मॉडल नाम या मॉडल का स्थानीय पथ पास किया जा सकता है। मूल्यांकन परिणामों को देखने के लिए `./output_dir/gpt2-xl_lmflow_chat_nll_eval`, `./output_dir/gpt2-xl_all_nll_eval`, और `./output_dir/gpt2-xl_commonsense_qa_eval` के अंतर्गत `benchmark.log` पर जा सकता है।


## Supported Features
<details> <summary>तेज़ प्रदर्शन और मेमोरी अनुकूलन के लिए फ़ाइन-ट्यूनिंग</summary>

* LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning
  
  LISA एक मेमोरी अनुकूल LLM माइक्रो-ट्यूनिंग एल्गोरिदम है। माइक्रो-ट्यूनिंग प्रक्रिया में लेयर को विशेष रूप से फ्रीज़ करके, LISA मौजूदा माइक्रो-ट्यूनिंग विधियों (जैसे LoRA) से आगे निकलता है। अधिक जानकारी के लिए [पेपर](https://arxiv.org/abs/2403.17919) पर जाएं।
  LISA का उपयोग करने के लिए प्रशिक्षण कमांड में पैरामीटर `--use_lisa 1` निर्दिष्ट किया जा सकता है। सक्रिय किए गए परतों की संख्या को `--lisa_activated_layers 2` द्वारा नियंत्रित किया जा सकता है, और फ्रीज़ की गई परतों के अंतराल को `--lisa_step_interval 20` द्वारा समायोजित किया जा सकता है।

* LoRA
  
  LoRA पैरामीटर-अनुकूल (parameter-efficient) माइक्रो-ट्यूनिंग एल्गोरिदम है, जो पूरे पैरामीटर माइक्रो-ट्यूनिंग से अधिक कुशल है। कृपया देखें: [माइक्रो-ट्यूनिंग (LoRA)](#fine-tuning-lora)।

* FlashAttention

  LMFlow में FlashAttention-1 और नवीनतम FlashAttention-2 दोनों का समर्थन है। अधिक जानकारी के लिए [flash_attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) देखें।

* Gradient Checkpointing

  [ग्रेडिएंट चेकपॉइंटिंग](https://github.com/cybertronai/gradient-checkpointing) एक मेमोरी अनुकूलन तकनीक है जो कंप्यूट को मेमोरी के लिए विनिमय करती है। 
  यह उपयोगी होता है जब मॉडल GPU मेमोरी में फिट करने के लिए बहुत बड़ा हो। 
  इसे आप अपने प्रशिक्षण कमांड में `--gradient_checkpointing` जोड़कर उपयोग करें।

* Deepspeed Zero3

  LMFlow [Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html) का समर्थन करता है। 
  हम एक उदाहरण [deepspeed कॉन्फ़िग](https://github.com/OptimalScale/LMFlow/blob/main/configs/ds_config_zero3.json) प्रदान करते हैं, और आप इसे सीधे उपयोग कर सकते हैं।

</details>

<details> <summary>अनुमान त्वरण</summary>

* LLaMA Inference on CPU

  [llama.cpp](https://github.com/ggerganov/llama.cpp) के महान प्रयासों के धन्यवाद। यह सभी के लिए संभव है कि उनके LLaMA मॉडलों को CPU पर 4-बिट क्वांटाइजेशन के साथ चलाया जाए। हम LLaMA LoRA वेट्स को `.pt` फ़ाइलों में रूपांतरित करने के लिए एक स्क्रिप्ट प्रदान करते हैं। आपको केवल llama.cpp में `convert-pth-to-ggml.py` का उपयोग करना होगा ताकि क्वांटाइजेशन किया जा सके।

* FlashAttention

  LMFlow दोनों FlashAttention-1 और नवीनतम FlashAttention-2 का समर्थन करता है। अधिक विवरण के लिए [flash_attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) देखें।

</details>

<details> <summary>लंबा संदर्भ</summary>

* LLaMA मॉडल के लिए स्थिति अंतर्पोलेशन

  अब एलएमफ्लो LMFlow नवीनतम लीनियर और NTK (न्यूरल कर्नेल सिद्धांत) स्केलिंग तकनीकों का समर्थन करता है। अधिक विवरण के लिए [पोज़िशन इंटरपोलेशन](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md) देखें।

</details>

<details> <summary>मॉडल कस्टमाइज़ेशन</summary>


* शब्दावली विस्तार

  अब आप अपने खुद के सेंटेंसपीस टोकनाइज़र को प्रशिक्षित कर सकते हैं और इसे मॉडल के मूल hf टोकनाइज़र के साथ मर्ज कर सकते हैं। अधिक विवरण के लिए [vocab_extension](https://github.com/OptimalScale/LMFlow/blob/main/scripts/vocab_extension) देखें।

</details>

<details> <summary>बहुविध</summary>

* Multimodal Chatbot

  एलएमफ्लो में चित्रों और पाठों के बहुसाधारण इनपुट का समर्थन है। हमारे [एलएमफ्लो बहुसाधारण चैटबॉट](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_vis_chatbot_gradio_minigpt4.sh) की जाँच करें।
  
</details>


## Support
यदि आपको किसी भी मदद की आवश्यकता हो तो, कृपया एक [Github](https://github.com/OptimalScale/LMFlow) इशु प्रस्तुत करें।


## License
इस परियोजना में शामिल कोड [Apache 2.0 लाइसेंस](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE) के तहत लाइसेंस प्राप्त है। इस परियोजना में शामिल कोड और मॉडल का व्यापारिक उद्देश्यों के लिए उपयोग करने की इच्छा हो तो, कृपया योगदानकर्ताओं से संपर्क करें।


## Citation
यदि आपको यह रेपो उपयोगी लगता है, तो कृपया ⭐ देने और उद्धरण करने का विचार करें:

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
