lm_eval --model hf \
    --model_args pretrained=nvidia/Hymba-1.5B-Instruct,trust_remote_code=True,cache_dir=~/.cache \
    --tasks elmb_functioncalling,elmb_chatrag,elmb_reasoning,elmb_roleplay \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/test_elmb