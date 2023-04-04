<p align="center" width="100%">
<img src="assets/logo.png" alt="LMFlow" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README_es.md">English</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README_zh-hans.md">简体中文</a> |
        <b>Español</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/README_jp.md">日本語</a>
    <p>
</h4>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Doc](https://img.shields.io/badge/Website-Doc-ff69b4.svg)](https://optimalscale.github.io/LMFlow/)
[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/srGxyazbNs)
[![WeChat badge](https://img.shields.io/badge/微信-加入-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)


Una caja de herramientas extensible, conveniente y eficiente para ajustar modelos de aprendizaje automático grandes, diseñada para ser fácil de usar, rápida, confiable y accesible para toda la comunidad.

Modelo de Lenguaje Grande para Todos. Vea nuestra [visión](https://github.com/OptimalScale/LMFlow#vision).


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

### Actualmente, nuestro servicio de descarga de checkpoints está en capacidad máxima. Hemos asignado un servidor adicional para apoyar esto. Si encuentras el error "demasiadas solicitudes HTTP", por favor espera varios minutos e intenta nuevamente. Gracias por tu comprensión. :pray:

Ofrecemos cuatro tipos de demostraciones que incluyen:

- Servicio en línea: Si no deseas ejecutar ningún código y simplemente quieres probar nuestros modelos, implementamos nuestros LLaMA-7B y LLaMA-33B ajustados con instrucciones para que puedas probarlos.
- Chatbot Colab (shell): Un chatbot interactivo basado en shell para que puedas implementar fácilmente un chatbot en Colab.
- Chatbot Colab (web): Un chatbot interactivo basado en web para que puedas implementar fácilmente tu propio chatbot en Colab.
- Implementación local: También ofrecemos una forma de implementar tu modelo/chatbot localmente, lo que significa que puedes implementar un modelo mucho más grande que los tres métodos anteriores si tienes suficientes recursos.


[![Code License](https://img.shields.io/badge/Online%20Service-Web-green.svg)](https://lmflow.com)
[![colab badge](https://img.shields.io/badge/Colab-(shell)%20%20chatbot:%20gpt--neo-orange?logo=google-colab&amp)](https://colab.research.google.com/drive/1P9Hf6_mLE7WHH92pw73j9D5kz6GTdkow?usp=sharing)
[![colab badge](https://img.shields.io/badge/Colab-(web)%20%20chatbot:%20gpt--neo-blue?logo=google-colab&amp)](https://colab.research.google.com/drive/1LLtiiQO-ZIIFsTKxYzGWYX9BDRc-v8dq?usp=sharing)


### Online Service
> Bienvenido/a a nuestro [servicio web](https://lmflow.com/). Tenemos desplegado en línea el modelo LLaMA-7B-tuned y LLaMA-33B-tuned para su vista previa. Debido al alto tráfico del sitio web, a veces puede que falle en responder. También puedes desplegar el chatbot haciendo referencia a `Local Deploy`.

### Colab chatbot(shell)
<p align="center" width="100%">
<img src="assets/colab-shell-chatbot-demo.png">
</p>

Proporcionamos una demostración simple de la línea de comandos del chatbot con T4/P100/V100 GPU de Google Colab. Es importante tener en cuenta que el modelo gpt-neo-2.7b proporcionado es un modelo bastante débil, que solo admite inglés y a veces puede generar respuestas insatisfactorias. Para mejorar su rendimiento, los usuarios pueden utilizar sus propios conjuntos de datos para ajustar y obtener un modelo mejor con LMFlow. También se pueden probar otros modelos solo decodificadores disponibles en 🤗 [huggingface](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).


```sh
./scripts/run_chatbot.sh {another-model-name}
```
### Colab chatbot(web)
Proporcionamos una demostración web simple del chatbot con T4/P100/V100 GPU de Google Colab. Es importante tener en cuenta que el modelo gpt-neo-2.7b proporcionado es un modelo bastante débil, que solo admite inglés y a veces puede generar respuestas insatisfactorias.



### Local Deploy
Si tienes recursos y quieres desplegar tu propio modelo localmente, te proporcionamos una forma sencilla de ejecutar un servidor Flask para lanzar la parte trasera (para proporcionar servicios adicionales a otras partes delantera) y una interfaz web interactiva (para permitirte comunicarte directamente) mediante:
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

El rendimiento de LLaMA 33B (LoRA) se logra con solo **~16h** de ajuste fino en la división de entrenamiento de PubMedQA y MedMCQA con un único servidor 8 * A100. Para obtener más rendimiento, incluidos los resultados del ajuste de instrucciones, consulta nuestra [documentación](https://optimalscale.github.io/LMFlow/).


## Model Zoo
Hemos hecho públicos los checkpoints entrenados para que todos puedan utilizarlos para un mayor entrenamiento e inferencia.

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

Ofrecemos soporte para todos los modelos [decodificadores](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) en 🤗 huggingface. Hemos probado completamente LLaMA, GPT2, GPT-Neo, Galactica. Pronto también ofreceremos soporte para modelos codificadores.

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
Puedes descargar fácilmente los conjuntos de datos de entrenamiento y prueba de ejemplo ejecutando:
```bash
cd data
bash download.sh all
cd -
``` 

También puedes utilizar tu propio conjunto de datos simplemente convirtiéndolo al siguiente formato:
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

Puedes ejecutar `scripts/run_finetune.sh` para ajustar finamente un modelo base GPT-2.
```sh
./scripts/run_finetune.sh
```

Si deseas proporcionar argumentos para DeepSpeed que reflejen la configuración de tu máquina, puedes pasar los argumentos correspondientes a DeepSpeed al script. Por ejemplo:
```sh
./scripts/run_finetune.sh "--num_gpus=8 --master_port 10001"
```

Para habilitar el ajuste fino de LoRA, puedes consultar:
```sh
./scripts/run_finetune_with_lora.sh
```
which can be run in similar manner.

Para obtener configuraciones detalladas, uno puede modificar estos scripts directamente. Estos scripts en realidad solo llaman al script de Python examples/finetune.py, que se puede ejecutar de la siguiente manera:

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

```python
python examples/finetune.py -h
```

### 3.2 Run Evaluation

Uno puede ejecutar la evaluación directamente con un modelo existente de Hugging Face, por ejemplo, para ejecutar GPT2 large, se puede ejecutar:
```sh
./scripts/run_evaluation.sh
```

Para modelos ajustados finamente con LoRA, uno puede consultar:
```sh
./scripts/run_evaluation_with_lora.sh
```

`tests`.
Esos scripts invocan los ejemplos `examples/*.py` construidos sobre nuestras APIs. Para obtener más ejemplos relacionados con las APIs, uno puede consultar los métodos en el unittest `tests`.


## 4. Additional Notes
### 4.1 LLaMA Checkpoint

1. Primero, necesitas obtener el acceso al modelo LLaMA desde [facebookresearch/llama](https://github.com/facebookresearch/llama). Descarga los checkpoints oficiales y guárdalos en ${llama-path}.

2. Segundo, convierte los checkpoints oficiales `${llama-path}` a checkpoints compatibles con HuggingFace $`${llama-hf-path}` ejecutando:

    `python ./scripts/convert_llama_weights_to_hf.py --input_dir ${llama-path} --model_size 7B --output_dir ${llama-hf-path}/llama-7b-hf`

3. ¡Listo! Ahora puedes establecer la ruta del checkpoint en `${llama-hf-path}/llama-7b-hf`. ¡Disfrútalo!


### 4.2 DeepSpeed Config

Puedes configurar DeepSpeed en configs. Los detalles se pueden consultar en [Configuración de DeepSpeed](https://www.deepspeed.ai/docs/config-json/).




## 5. Model Release

### 5.1 Medical Model Checkpoints
Puedes ejecutar el siguiente script para descargar los checkpoints de nuestro modelo médico::

```bash
cd output_models
bash download.sh medical_ckpt
cd -
```
También puedes descargar directamente nuestro modelo a través del enlace de Google Drive: [medical_ckpt.tar.gz](https://drive.google.com/file/d/1bnsQGNGNYchsOfiNyRAmL2fNiowbmFNw/view?usp=share_link)

### 5.2 Instruction Model Checkpoints
Similarly, you can run following script to download our instruction model checkpoints :
```bash
cd output_models
bash download.sh instruction_ckpt
cd -
```

Por supuesto, ¿podrías proporcionarme el enlace de Google Drive para que pueda ayudarte a descargar el modelo directamente: [instruction_ckpt.tar.gz](https://drive.google.com/file/d/1d_ioQ-ViVweeifbsFSO4pczc3UORFHZO/view?usp=share_link)

### 5.3 Begin Reproduce
Después de descargar los checkpoints del modelo, puedes reemplazar `--lora_model_path` con `output_models/instruction_ckpt/llama7b-lora` (ejemplo para llama-7b para instrucciones) y reemplazar `--model_name_or_path` con tu modelo LLaMA convertido dentro de `LMFlow/scripts/run_evaluation_with_lora.sh` y ejecutar este script de shell para reproducir el resultado.

Luego puedes verificar el rendimiento del modelo en nuestra [documentación](https://optimalscale.github.io/LMFlow/).


## Documentation
Por favor, consulta nuestra [documentación](https://optimalscale.github.io/LMFlow/) para obtener más referencias a la API y resultados experimentales.


## Vision
¡Hola! ¡Estamos emocionados de anunciar el próximo lanzamiento de nuestro repositorio de código que incluye un proceso completo de entrenamiento de LLM, permitiendo a los usuarios construir rápidamente sus propios modelos de lenguaje y entrenarlos de manera efectiva!

Nuestro repositorio de código no es solo un modelo simple; incluye todo el flujo de trabajo de entrenamiento, optimización de modelo y herramientas de prueba. Puedes usarlo para construir varios tipos de modelos de lenguaje, incluyendo modelos de conversación, modelos de pregunta-respuesta y modelos de generación de texto, entre otros.

Además, nuestro objetivo es crear una plataforma abierta y democrática de intercambio de LLM donde las personas puedan compartir sus checkpoints y experiencias para mejorar colectivamente las habilidades de la comunidad. ¡Damos la bienvenida a cualquier persona interesada en LLM a participar y unirse a nosotros en la construcción de una comunidad abierta y amigable!

Ya seas principiante o experto, creemos que puedes beneficiarte de esta plataforma. ¡Trabajemos juntos para construir una comunidad vibrante e innovadora de LLM!

[![Embark](https://img.shields.io/badge/discord-LMFlow-%237289da.svg?logo=discord)](https://discord.gg/srGxyazbNs)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/04/04/ibvpAk.jpeg)


## Disclaimer
Este paquete tiene como objetivo proporcionar un flujo de trabajo optimizado y fácil de usar para el ajuste de modelos grandes. Sus funcionalidades sirven como referencia y están destinadas a ser utilizadas por el usuario. Sin embargo, es importante tener en cuenta que la responsabilidad por la preparación de los datos y modelos pre-entrenados recae únicamente en el usuario. Este paquete no garantiza la precisión, integridad, aplicabilidad o legalidad de los componentes provenientes de la preparación del usuario. Los usuarios deben ser conscientes y asumir todos los riesgos y responsabilidades asociados con la preparación de los modelos y datos, y obtener asesoramiento legal, comercial y técnico antes de utilizar este paquete. El flujo de trabajo no será responsable de ningún daño directo, indirecto, especial, incidental o consecuente resultante de la preparación indebida de los datos y modelos pre-entrenados por parte del usuario.

Nuestros checkpoints, que incluyen versiones en inglés y chino, se proporcionan únicamente con fines de investigación. Los datos de entrenamiento contenidos en estos checkpoints incluyen resultados generados a partir del modelo de lenguaje ChatGPT. No respaldamos ni fomentamos la distribución o uso de estos checkpoints con fines comerciales. Los usuarios de estos checkpoints son responsables únicamente de garantizar que se utilicen correctamente y de forma apropiada.

También es crucial destacar que los resultados generados por el modelo se basan en modelos probabilísticos y no están directamente relacionados con este paquete. La precisión, confiabilidad, aplicabilidad y legalidad de los resultados no están garantizados por este paquete. Por lo tanto, los usuarios también deben ser conscientes de los riesgos y responsabilidades asociados con los resultados y buscar asesoramiento legal, comercial y técnico antes de confiar en los resultados generados por el modelo. Este paquete no será responsable de ningún daño directo, indirecto, especial, incidental o consecuente resultante de la dependencia de los resultados generados por el modelo por parte del usuario.

## Support

If you need any help, please submit a [Github](https://github.com/OptimalScale/LMFlow) issue.

## Contributors
<a href="https://github.com/OptimalScale/LMFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OptimalScale/LMFlow" />
</a>

## Citation
Si encuentras este repositorio útil, por favor considera darle ⭐ y citarlo:

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


