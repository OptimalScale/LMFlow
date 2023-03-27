common_args="temperature=0.0,do_sample=False,min_length=5,max_length=100"

# excute old old_inference.py  commit_id: 4574d3d646b4fa7b4aa42689c3e6f4c0b169703f HEAD -> shizhe-regression-test
deepspeed --num_gpus=4 inference.py --dataset medmcqa-regression --model_name gpt2-medium --model_args ${common_args}

# excute new new_inference.py
python examples/inference.py --model_args ${common_args} -> generate new_output.log

# do comparison
diff old_output.log new_output.log