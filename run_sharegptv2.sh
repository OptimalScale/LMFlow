

num_train_epochs=0.0001

exp_name="069_sharegptv1_vicuna7b_lora_5epcoh_lr1e-4"; 
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10069 --num_gpus=8" "--preprocessing_num_workers 80 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v2/train" 

./scripts/run_evaluation_ppl_cn.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

exp_name="070_sharegptv1_vicuna7b_lora_5epcoh_lr5e-5";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10070 --num_gpus=8" "--preprocessing_num_workers 80 --model_name_or_path /home/home/Tribbiani/vicuna-7b  --num_train_epochs ${num_train_epochs} --learning_rate 5e-5 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v2/train"

./scripts/run_evaluation_ppl_cn.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh /home/home/Tribbiani/vicuna-7b output_models/${exp_name} ${exp_name}

exp_name="071_sharegptv1_llama7b_lora_5epcoh_lr1e-4";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10071 --num_gpus=8" "--preprocessing_num_workers 80 --model_name_or_path pinkmanlove/llama-7b-hf  --num_train_epochs ${num_train_epochs} --learning_rate 1e-4 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v2/train"

./scripts/run_evaluation_ppl_cn.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}

exp_name="072_sharegptv1_llama7b_lora_5epcoh_lr5e-5";
./scripts/run_finetune_with_lora.sh ${exp_name} "--master_port=10072 --num_gpus=8" "--preprocessing_num_workers 80 --model_name_or_path pinkmanlove/llama-7b-hf  --num_train_epochs ${num_train_epochs} --learning_rate 5e-5 --per_device_train_batch_size 3 --block_size 768 --lora_r 8 --dataset_path data/sharegpt_v2/train"

./scripts/run_evaluation_ppl_cn.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}
./scripts/run_evaluation_ppl_en.sh pinkmanlove/llama-7b-hf output_models/${exp_name} ${exp_name}
