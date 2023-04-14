

export WANDB_MODE=disabled
num_train_epochs=5

exp_name="065_sharegpt_v8_vicuna7b_lora_5epcoh_lr1e-4"; 
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10065 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v8/train" 

./scripts/run_evaluation_ppl_cn.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

exp_name="066_sharegpt_v8_vicuna7b_lora_5epcoh_lr5e-5";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10066 --num_gpus=8" "--preprocessing_num_workers 40 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 5e-5 --per_device_train_batch_size 2 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v8/train"

./scripts/run_evaluation_ppl_cn.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

