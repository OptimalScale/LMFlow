## Function-call Finetune

### Pip dependency

```
bitsandbytes==0.40.0
deepspeed==0.12.0
flash-attn==2.5.7
peft==0.10.0
torch==2.1.2+cu118
transformers==4.40.1
vllm==0.5.2
xformers==0.0.27
```

### Conversation Template
```
{
    "type": "conversation",
    "instances": [
        {
            "system": "You are a helpful assistant with access to the following functions. Use them if required - ",
            "tools": ["{\"name\": \"", \"description\": \"", \"parameters\": {\"type\": \"object\", \"properties\": {\"property_1\": {\"type\": \"xxx\", \"description\": \"\"}, \"property_2\": {\"type\": \"xxx\", \"description\": \"\"}}, \"required\": [\"required_1\", \"property_n\"]}}",]",
            "messages": [
                {
                    "role": "user",
                    "content": ""
                },
                {
                    "role": "function",
                    "content": ""
                },
                {
                    "role": "observation",
                    "content": ""
                },
                {
                    "role": "assistant",
                    "content": ""
                }
            ]
        },
        {
            "system": "You are a helpful assistant, with no access to external functions.",
            "tools": [],
            "messages": [
                {
                    "role": "user",
                    "content": ""
                },
                {
                    "role": "assistant",
                    "content": ""
                }
            ]
        },
    ]
}
```

### Run Function-call Finetune Example
```
./contrib/tool-finetune/run_function_call_finetune.sh \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --trust_remote_code True \
    --conversation_template llama3_for_tool \
    --dataset_path /home/wenhesun/LMFlow/data/glaive-function-calling-v2 \
    --output_model_path /home/wenhesun/LMFlow/output_models/function-call-finetuned-llama
```

### Command-Line Arguments
- `--model-name-or-path` - Specifies the name or path of the model used for
- `--conversation_template` - So far supports the following choices: llama3_for_tool, qwen2_for_tool
- `--dataset_path` - The path to the dataset that has been converted to the specified format
- `--output_model_path` - Directory to store the finetuned model and logs