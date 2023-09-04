# MultiModal Conversation
## Pretrain
First following [LLava](!https://github.com/haotian-liu/LLaVA) to download the dataset and run the following script for pretraining:
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