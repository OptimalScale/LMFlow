# MultiModal Conversation
## Download dataset
We use the dataset from LLava to present the example of multi-modaltiy training.
Please first download the [pretrain dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) for pre-training. Then download the [coco 2017](https://cocodataset.org/) and the [conversation file](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_80k.json) for finetuning.
After downloading, modify the data path in the training script to your own path.
## Pretrain
Run the following script for pretraining:
```
bash scripts/multimodal/run_finetune_multi_modal_stage1.sh
``` 

## Finetune
Modify the path of the dataset and the pretrain language projection model and run the following script:
```
bash script/multimodal/run_finetune_multi_modal_stage2.sh
```

## Inference on CLI
Run the following script for LLava:
```
bash script/multimodal/run_vis_chatbot_llava.sh

```

Run the following script for mini gpt-4:
```
bash script/multimodal/run_vis_chatbot_minigpt4.sh
```

## Inference on gradio:
Run the following script for mini gpt-4:
```
bash script/multimodal/run_vis_chatbot_gradio_minigpt4.sh
```