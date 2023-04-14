

export WANDB_MODE=disabled
num_train_epochs=5

# exp_name="082_sharegpt_cleaned_full_QA_v2_vicuna7b_lora_5epcoh_lr1e-4"; 
# ./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10082 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_cleaned_full_QA_v2/train" 

# ./scripts/run_evaluation_ppl_cn_full.sh output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en_full.sh output_models/${exp_name} ${exp_name}

exp_name="083_sharegpt_small_mixed_gpt4_QA_vicuna7b_lora_5epcoh_lr1e-4";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10083 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_small_mixed_gpt4_QA/train"

# ./scripts/run_evaluation_ppl_cn_full.sh output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en_full.sh output_models/${exp_name} ${exp_name}

# exp_name="084_sharegpt_cleaned_small_QA_v1_vicuna7b_lora_5epcoh_lr1e-4";
# ./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10084 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_cleaned_small_QA_v1/train"

# ./scripts/run_evaluation_ppl_cn_full.sh output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en_full.sh output_models/${exp_name} ${exp_name}

# exp_name="085_sharegpt_cleaned_full_HA_v2_vicuna7b_lora_5epcoh_lr1e-4"; 
# ./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10085 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_cleaned_full_HA_v2/train" 

# ./scripts/run_evaluation_ppl_cn_full.sh output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en_full.sh output_models/${exp_name} ${exp_name}

# exp_name="086_sharegpt_small_mixed_gpt4_HA_vicuna7b_lora_5epcoh_lr1e-4";
# ./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10086 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_small_mixed_gpt4_HA/train"

# ./scripts/run_evaluation_ppl_cn_full.sh output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en_full.sh output_models/${exp_name} ${exp_name}

# exp_name="087_sharegpt_cleaned_small_HA_v1_vicuna7b_lora_5epcoh_lr1e-4";
# ./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10087 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_cleaned_small_HA_v1/train"

# ./scripts/run_evaluation_ppl_cn_full.sh output_models/${exp_name} ${exp_name}
# ./scripts/run_evaluation_ppl_en_full.sh output_models/${exp_name} ${exp_name}