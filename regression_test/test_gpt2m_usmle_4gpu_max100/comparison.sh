common_args="temperature=0.0,do_sample=False,min_length=5,max_length=100"
python examples/inference.py --model_args  > regression_test/test_gpt2m_usmle_4gpu_max100/new_output.log
