output_dir=output_models/rewardmodeling

--deepspeed script_args.deepspeed \
--local_rank script_args.local_rank \

--do_train True \
--output_dir ${output_dir}  \
--learning_rate 1e-5 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--num_train_epochs 1 \
--weight_decay 0.001 \
--evaluation_strategy "steps" \
--eval_steps 999999 \
--save_strategy "steps" \
--save_steps 999999 \
--gradient_accumulation_steps 32 \
--gradient_checkpointing True \
--remove_unused_columns False \
--bf16 True \
--logging_strategy "steps" \
--logging_steps 10 \
--optim "paged_adamw_32bit" \
--lr_scheduler_type "cosine" \
--warmup_ratio 0.03 \
--report_to 'wandb'
