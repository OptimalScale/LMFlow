common_args="temperature=0.0,do_sample=False,min_length=5,max_length=100"

# excute old old_inference.py  commit_id: d5fecf30ba8011067b10cf51fede53a5ab6574e4 HEAD -> shizhe-regression-test
deepspeed --num_gpus=1 inference.py --dataset medmcqa-regression --model_name gpt2 --model_args ${common_args}

# excute new new_inference.py
python examples/inference.py --model_args ${common_args} -> generate new_output.log

# do comparison
diff old_output.log new_output.log