

num_train_epochs=0.0001

exp_name="065_sharegptv1_vicuna7b_lora_5epcoh_lr1e-4"; 
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10065 --num_gpus=8" "--preprocessing_num_workers 128 --model_name_or_path /root/autodl-tmp/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v1/train" 

./scripts/run_evaluation_ppl_cn.sh /root/autodl-tmp/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh /root/autodl-tmp/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

exp_name="066_sharegptv1_vicuna7b_lora_5epcoh_lr5e-5";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10066 --num_gpus=8" "--preprocessing_num_workers 128 --model_name_or_path /root/autodl-tmp/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 5e-5 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v1/train"

./scripts/run_evaluation_ppl_cn.sh /root/autodl-tmp/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh /root/autodl-tmp/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

exp_name="067_sharegptv1_llama7b_lora_5epcoh_lr1e-4";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10067 --num_gpus=8" "--preprocessing_num_workers 128 --model_name_or_path pinkmanlove/llama-7b-hf  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v1/train"

./scripts/run_evaluation_ppl_cn.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}

exp_name="068_sharegptv1_llama7b_lora_5epcoh_lr5e-5";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10068 --num_gpus=8" "--preprocessing_num_workers 128 --model_name_or_path pinkmanlove/llama-7b-hf  --num_train_epochs ${num_train_epochs} --learning_rate 5e-5 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v1/train"

./scripts/run_evaluation_ppl_cn.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}
