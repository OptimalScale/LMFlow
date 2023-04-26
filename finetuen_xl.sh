wandb offline
lr=2e-5
bs=4
use_lora=1
epochs=3
gradient_checkpointing=False
gradient_accumulation_steps=4
lora_r=32
ds_config=configs/ds_config_zero2.json
model_name_or_path=pinkmanlove/llama-7b-hf
exp_name="xl_050";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_HA"
data_path="/home/xiangliu/LMFlow/data/example_dataset/train"
eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} "--master_port=10065 --num_gpus=8" 

bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 
