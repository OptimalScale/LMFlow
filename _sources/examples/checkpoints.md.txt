# Checkpoints

In general, you can directly load from checkpoints by using `--model_name_or_path`. However, the LLaMA case is slightly different due to the copyright issue.


## LLaMA Checkpoint

1. First, you need to get the access of LLaMA model from [facebookresearch/llama](https://github.com/facebookresearch/llama). Download the official checkpoints and save them into `${llama-path}`.

2. Second, convert the official checkpoints `${llama-path}` to HuggingFace supported checkpoints `${llama-hf-path}` by running

    `python ./scripts/convert_llama_weights_to_hf.py --input_dir ${llama-path} --model_size 7B --output_dir ${llama-hf-path}/llama-7b-hf`

3. Then you are good to go by setting the checkpoint path to `${llama-hf-path}/llama-7b-hf`. Enjoy it!

4. (optional) Now you have the original llama-7b-hf pretrained model. With
```sh
cd output_models && ./download.sh all && cd -
```
You can obtain the model difference finetuned by ours. By a way similar to `./scripts/run_evaluation_with_lora.sh`,
```sh
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text \
    --model_name_or_path ${llama-hf-path}/llama-7b-hf \
    --lora_model_path output_models/${llama-model-diff-path} \
    --dataset_path data/alpaca/test \
    --prompt_structure "Input: {input}" \
    --deepspeed examples/ds_config.json
```
You can now evaluate with the finetuned llama model.