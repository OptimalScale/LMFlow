common_args="temperature=0.0,do_sample=False,min_length=5,max_length=100"

# excute old old_inference.py  commit_id: d95babb0d9e5667dd7d822947f169c16a87079b7 HEAD -> shizhe-regression-test
deepspeed --num_gpus=4 inference.py --dataset gsm8k-regression --model_name gpt2-medium --model_args ${common_args}

# excute new new_inference.py
python examples/inference.py --model_args ${common_args} -> generate new_output.log

# do comparison
diff old_output.log new_output.log