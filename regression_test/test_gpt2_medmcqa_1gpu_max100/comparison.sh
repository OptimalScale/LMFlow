common_args="temperature=0.0,do_sample=False,min_length=5,max_length=100"
python examples/inference.py --model_args  > regression_test/test_gpt2_medmcqa_1gpu_max100/new_output.log
