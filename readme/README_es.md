<p align="center" width="100%">
<img src="../assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README.md">English</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_zh-hans.md">简体中文</a> |
        <b>Español</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_jp.md">日本語</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_ko.md">한국어</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_hindi.md">हिंदी</a>
    <p>
</h4>

> [!NOTE]
> This README file was translated by LLM. Spanish speakers are welcome to submit PRs to polish the document!  

> [!NOTE]  
La versión en español fue traducida por ChatGPT, si hay algún error, bienvenido sea al contributor para corregirlo, gracias. Al mismo tiempo, si hay alguna diferencia o inconsistencia en el contenido con la versión en inglés, se debe considerar la versión en inglés como la correcta.

[![Website](https://img.shields.io/badge/Website-Demo-20B2AA.svg)](https://lmflow.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/Discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1wju9nicy-woXbNtS~5MavHSAtiMxmxQ)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://ibb.co/ZhM4hhn)

Una caja de herramientas extensible, conveniente y eficiente para ajustar modelos de aprendizaje automático grandes, diseñada para ser fácil de usar, rápida, confiable y accesible para toda la comunidad.


<p align="center" width="100%">
<img src="../assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Latest News
* [2024-04-25] :rocket: ¡Soporte para plantilla de conversación! Hemos preconfigurado las últimas plantillas de conversación [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B) y [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct), así como algunas plantillas de conversación frecuentemente utilizadas como `chatml` (ver todas las plantillas [aquí](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#conversation-template)), y estamos trabajando en agregar más plantillas preconfiguradas. ¡Agrega el correspondiente `--conversation_template` en el script de la terminal y estarás listo! :rocket:  
* [2024-03-27] Soporte para [LISA](https://arxiv.org/abs/2403.17919) — ¡Entrenamiento de modelos de 7B en GPU con 24G de memoria sin necesidad de offloading!  
* [2023-09-11] Soporte para [decodificación especulativa](https://arxiv.org/abs/2211.17192), consulta la [guía de uso](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md) para ver cómo utilizarlo y estadísticas de rendimiento básicas.
* [2023-08-14] Soporte para ampliar la ventana de contexto de LLaMA a través de interpolación de posición (Lineal y Escalado NTK), más información en: [Interpolación de Posición](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md).
* [2023-08-07] Soporte para [Flash Attention-2](https://crfm.stanford.edu/2023/07/17/flash2.html), consulta la [guía de uso de Flash Attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) para más detalles.


## Quick Start
### Setup
Nuestro repositorio ha sido probado en Linux (Ubuntu 20.04). Las otras plataformas de sistemas operativos (macOS, Windows) aún no han sido completamente probadas, por lo que pueden surgir algunos errores inesperados. Se recomienda probar primero en Linux/Windows WSL o utilizar Google Colab para experimentar.

Para CUDA 10.3-11.7, se recomienda utilizar `v0.0.5` o versiones anteriores. Para CUDA superior a 11.7, por favor, utilice nuestra rama estable `>= v0.0.6` para una mejor experiencia.
```bash
git clone https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```

### Prepare Dataset
Por favor, consulta nuestra [documentación oficial (en inglés)](https://optimalscale.github.io/LMFlow/examples/DATASETS.html). La documentación oficial se encuentra actualmente en proceso de traducción, te pedimos paciencia mientras tanto.

### Fine-Tuning (Full)
El ajuste fino completo actualizará todos los parámetros del modelo. A continuación se muestra un ejemplo de ajuste fino completo de GPT-2:

```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

> [!TIP]
> Puedes especificar una plantilla de conversación para el conjunto de datos de diálogo agregando el parámetro `--conversation_template`.
>
><details><summary>Ejemplo: Especificar una plantilla de conversación para Llama-3-8B</summary>  
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
[LISA](https://arxiv.org/abs/2403.17919) es un algoritmo de ajuste fino que es **eficiente en memoria**, permitiendo un equilibrio entre la memoria y el número de capas descongeladas aleatoriamente. El script siguiente ha sido probado únicamente en **una sola GPU**. ¡Estén atentos a nuestras últimas actualizaciones! :smile:

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
> <details><summary>Ejemplo: Especificando el conjunto de datos de conversación para Llama-2-7B</summary>  
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
LoRA es un algoritmo de ajuste fino de parámetros que es más eficiente que el ajuste fino completo de parámetros.
```sh
cd data && ./download.sh alpaca && cd -

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path facebook/galactica-1.3b \
  --dataset_path data/alpaca/train_conversation \
  --output_lora_path output_models/finetuned_galactica_lora
```

> [!TIP]
> <details><summary>Ejemplo: Especificando el conjunto de datos de diálogo para Llama-2-7B</summary>  
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
> <details><summary>Combinando pesos de LoRA</summary>
>
>Puede combinar los pesos de LoRA con el modelo original utilizando el siguiente comando:  
>```sh
>./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>

### Inference
Después de haber terminado el ajuste fino, puedes entablar una conversación con el modelo usando el siguiente comando.
```sh
./scripts/run_chatbot.sh output_models/finetuned_gpt2
```

### Deployment
Si deseas implementar tu propio modelo localmente, ofrecemos una interfaz de chatbot basada en Gradio.
Para iniciar la demostración de Robin-7b con esta interfaz, utilice los siguientes comandos:
```sh
pip install gradio
python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path YOUR-LLAMA  --lora_model_path ./robin-7b --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:"       --end_string "#" --max_new_tokens 200
```

### Evaluation
[LMFlow Benchmark](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418) es un marco de evaluación automática para LLM de código abierto. Utilizamos la Probabilidad Negativa del Logaritmo (NLL) como métrica para evaluar diversos aspectos de los LLM, como el chat casual, el razonamiento común y la capacidad de seguir instrucciones. Le invitamos a utilizar LMFlow Benchmark para evaluar los modelos que tenga disponibles y a participar en nuestra [Comparación de Modelos (LLM comparision)](https://docs.google.com/spreadsheets/d/1JYh4_pxNzmNA9I0YM2epgRA7VXBIeIGS64gPJBg5NHA/edit?usp=sharing).

Tomando como ejemplo el GPT-2 XL, puede comenzar la evaluación con el siguiente comando:
```sh
./scripts/run_benchmark.sh --model_name_or_path gpt2-xl
```
`--model_name_or_path` es un parámetro obligatorio, donde puede ingresar el nombre del modelo de Hugging Face o la ruta local del modelo.
Puede revisar los resultados de la evaluación en `benchmark.log` dentro de `./output_dir/gpt2-xl_lmflow_chat_nll_eval`, `./output_dir/gpt2-xl_all_nll_eval` y `./output_dir/gpt2-xl_commonsense_qa_eval`.


## Supported Features
<details> <summary>Optimización de Ajuste Fino y Memoria</summary>

* LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning
  
  LISA es un algoritmo de ajuste fino de LLM eficiente en memoria. Al seleccionar selectivamente capas para congelar durante el ajuste fino, LISA supera los métodos de ajuste fino existentes (como LoRA). Consulta el [documento](https://arxiv.org/abs/2403.17919) para obtener más información. Puedes utilizar LISA especificando el parámetro `--use_lisa 1` en el comando de entrenamiento. Controla el número de capas activadas con `--lisa_activated_layers 2` y ajusta el intervalo de congelación de capas con `--lisa_step_interval 20`.

* LoRA
  
  LoRA es un algoritmo de ajuste fino eficiente en parámetros que es más eficiente que el ajuste fino de todos los parámetros. Consulta [Ajuste Fino (LoRA)](#fine-tuning-lora) para más detalles.

* FlashAttention
  
  Soportamos FlashAttention-1 y FlashAttention-2. Para más detalles, consulta: [FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md).

* Punto de Control de Gradientes
  
  [Punto de control de gradientes](https://github.com/cybertronai/gradient-checkpointing) es una técnica de optimización de memoria que intercambia cálculos por memoria para reducir el uso de la memoria de la GPU. Puedes utilizarlo agregando `--gradient_checkpointing` al comando de entrenamiento.

* Deepspeed Zero3
  
  LMFlow es compatible con [Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html). Proporcionamos un archivo de configuración de deepspeed listo para usar [aquí](https://github.com/OptimalScale/LMFlow/blob/main/configs/ds_config_zero3.json).

</details>


<details> <summary>Aceleración de inferencia</summary>

* Inferencia de CPU LLaMA
  
  ¡Gracias a [llama.cpp](https://github.com/ggerganov/llama.cpp), ahora todos pueden ejecutar su propio LLaMA (cuantificación de 4 bits) en la CPU! Proporcionamos un script para convertir los pesos de LLaMA LoRA en archivos `.pt`, solo necesita usar `convert-pth-to-ggml.py` de llama.cpp para realizar la cuantificación del modelo y así realizar la inferencia de LLaMA en la CPU.

* FlashAttention
  
  Apoyamos FlashAttention-1 y FlashAttention-2. Para más detalles, consulta: [FlashAttention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md).

</details>


<details> <summary>Texto largo</summary>

* Interpolación de posición del modelo LLaMA (Position Interpolation)
  
  Se admite la extensión del contexto de la ventana LLaMA mediante interpolación de posición (Position Interpolation) (escalamiento lineal y NTK), consulte más detalles en: [Interpolación de posición](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md).

</details>


<details> <summary>Personalización del modelo</summary>

* Ampliación del vocabulario
  
  Entrena tu propio tokenizador de SentencePiece y luego combínalo con el tokenizador de Hugging Face que viene con el modelo. Consulta: [Ampliación del vocabulario](https://github.com/OptimalScale/LMFlow/blob/main/scripts/vocab_extension).

</details>


<details> <summary>Multi-modal</summary>

* Chatbot multi-modal
  
  LMFlow admite entradas multi-modales (imágenes, texto). Consulta: [Chatbot multi-modal de LMFlow](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_vis_chatbot_gradio_minigpt4.sh).

</details>


## Support
Si necesitas ayuda, no dudes en presentar un [problema en Github](https://github.com/OptimalScale/LMFlow/issues).


## License
El código incluido en este proyecto está bajo la licencia Apache 2.0. Si desea utilizar los modelos incluidos en este proyecto para fines comerciales, por favor, póngase en contacto con el desarrollador para obtener autorización.


## Citation
Si encuentras este repositorio útil, por favor considera darle ⭐ y citarlo:

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