common_args="temperature=0.0,do_sample=False,min_length=5,max_length=50"
python examples/inference.py --model_args  > regression_test/test_gpt2_medmcqa_2gpu_max50/new_output.log
