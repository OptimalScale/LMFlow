wandb offline
lr=8e-4
bs=4
use_lora=0
epochs=10
gradient_checkpointing=False
gradient_accumulation_steps=4
lora_r=32
ds_config=configs/ds_config_zero2.json
model_name_or_path=gpt2
exp_name="test";
# data_path="/home/xiangliu/LMFlow/data/cn_v1_HA"
data_path="/home/xiangliu/LMFlow/data/example_dataset/train"
eval_dataset_path="/home/xiangliu/LMFlow/data/gpt4_eval"
bash ./scripts/run_finetune_with_lora_save_aggregated_weights.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} "--master_port=10065 --num_gpus=1" 

