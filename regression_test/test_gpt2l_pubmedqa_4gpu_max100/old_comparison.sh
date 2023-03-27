common_args="temperature=0.0,do_sample=False,min_length=5,max_length=100"

# excute old old_inference.py  commit_id: 9a55bec911d7566cb4db6d89d385bc236f91e48e HEAD -> shizhe-regression-test
deepspeed --num_gpus=4 inference.py --dataset pubmedqa-regression --model_name gpt2-large --model_args ${common_args}

# excute new new_inference.py
python examples/inference.py --model_args ${common_args} -> generate new_output.log

# do comparison
diff old_output.log new_output.log