#!/usr/bin/env python3
"""
Script to upload a PEFT (DoRA) finetuned model and adapter to the Hugging Face Hub
"""
import argparse
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login, HfApi, create_repo

def parse_args():
    parser = argparse.ArgumentParser(description="Upload PEFT model to Hugging Face Hub")
    parser.add_argument("--base_model", type=str,
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str,
                        help="Path to DoRA adapter model")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="Hugging Face API token")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Repository ID for the uploaded model (format: username/model-name)")
    parser.add_argument("--adapter_repo_id", type=str,
                        help="Repository ID for the adapter only (format: username/adapter-name). If not provided, will use repo_id + '-adapter'")
    parser.add_argument("--description", type=str,
                        help="Short description for the model card")
    parser.add_argument("--merge_weights", action="store_true",
                        help="Merge LoRA weights with base model before uploading")
    parser.add_argument("--upload_adapter", action="store_true",
                        help="Whether to upload the adapter separately")
    parser.add_argument("--model_type", type=str,
                        help="Model type to set in config.json. If not provided, will try to extract from base model")
    return parser.parse_args()

def main():
    args = parse_args()
    # Log in to Hugging Face
    login(token=args.hf_token)
    api = HfApi()
    # Set adapter repo ID if not provided
    if not hasattr(args, 'adapter_repo_id') or args.adapter_repo_id is None:
        args.adapter_repo_id = f"{args.repo_id}-adapter"
    print(f"Loading base model: {args.base_model}")
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    # Try to get the model_type from base model config if not provided
    if not args.model_type:
        if hasattr(base_model.config, 'model_type'):
            args.model_type = base_model.config.model_type
            print(f"Extracted model_type '{args.model_type}' from base model")
        else:
            # Try to infer from model name
            base_model_name = args.base_model.split("/")[-1].lower()
            if "llama" in base_model_name:
                args.model_type = "llama"
            elif "t5" in base_model_name:
                args.model_type = "t5"
            elif "bert" in base_model_name:
                args.model_type = "bert"
            else:
                args.model_type = "auto"
            print(f"Inferred model_type '{args.model_type}' from model name")
    print(f"Loading adapter from: {args.adapter_path}")
    # Load DoRA weights
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    # Quick test to ensure model works
    input_text = "What is the capital of France?"
    inputs = tokenizer(input_text, return_tensors="pt")
    print("Testing model with a simple prompt...")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=50)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")
    # Upload adapter separately if requested
    if args.upload_adapter:
        # First create the repository for the adapter
        print(f"Creating repository for adapter: {args.adapter_repo_id}")
        try:
            create_repo(
                repo_id=args.adapter_repo_id,
                token=args.hf_token,
                repo_type="model",
                exist_ok=True
            )
        except Exception as e:
            print(f"Error creating adapter repository: {e}")
            print("Continuing with model upload...")
        else:
            # Create a detailed adapter model card
            adapter_model_card = f"""---
language:
- en
tags:
- llama
- peft
- dora
- lora
- adapter
license: apache-2.0
base_model: {args.base_model}
---
# {args.adapter_repo_id.split('/')[-1]}
Adapter only for {args.description}
## Adapter Details
This is the DoRA adapter for [{args.repo_id}](https://huggingface.co/{args.repo_id}).
## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# Load the base model first
base_model = AutoModelForCausalLM.from_pretrained("{args.base_model}")
# Load the DoRA adapter
model = PeftModel.from_pretrained(base_model, "{args.adapter_repo_id}")
# Load the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained("{args.base_model}")
# Example usage
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```"""
            # Create README.md in adapter directory
            readme_path = os.path.join(args.adapter_path, "README.md")
            with open(readme_path, "w") as f:
                f.write(adapter_model_card)
            # Upload adapter to Hub
            try:
                api.upload_folder(
                    folder_path=args.adapter_path,
                    repo_id=args.adapter_repo_id,
                    repo_type="model",
                    commit_message="Upload DoRA adapter",
                )
                print(f"Adapter uploaded successfully to {args.adapter_repo_id}")
            except Exception as e:
                print(f"Error uploading adapter: {e}")
                print("Continuing with model upload...")
    # Now handle the main model
    if args.merge_weights:
        print("Merging DoRA weights with base model...")
        model = model.merge_and_unload()
        print("Weights merged successfully")
    # Create repository for the main model
    print(f"Creating repository for model: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.hf_token,
            repo_type="model",
            exist_ok=True
        )
    except Exception as e:
        print(f"Error creating model repository: {e}")
        print("Repository might already exist, continuing...")
    # Create a detailed model card that includes links to the adapter if it was uploaded
    adapter_link = f"The standalone adapter is available at [{args.adapter_repo_id}](https://huggingface.co/{args.adapter_repo_id})." if args.upload_adapter else ""
    adapter_usage = f"""# Option 2: Load just the adapter with the base model
base_model = AutoModelForCausalLM.from_pretrained("{args.base_model}")
tokenizer = AutoTokenizer.from_pretrained("{args.base_model}")
model = PeftModel.from_pretrained(base_model, "{args.adapter_repo_id}")""" if args.upload_adapter else ""
    model_card = f"""---
language:
- en
tags:
- llama
- peft
- dora
- lora
license: apache-2.0
base_model: {args.base_model}
---
# {args.repo_id.split('/')[-1]}
{args.description}
## Model Details
This model is a DoRA-finetuned version of [{args.base_model}](https://huggingface.co/{args.base_model}).
{adapter_link}
## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# Option 1: Load the complete model directly
model = AutoModelForCausalLM.from_pretrained("{args.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")
{adapter_usage}
# Example usage
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```"""
    # Push model to Hugging Face Hub
    print(f"Uploading {'merged' if args.merge_weights else 'PEFT'} model to {args.repo_id}...")
    try:
        # IMPORTANT: Ensure model_type is set in config
        if hasattr(model, 'config'):
            if not hasattr(model.config, 'model_type') or not model.config.model_type:
                print(f"Setting model_type to '{args.model_type}' in config")
                model.config.model_type = args.model_type
        # Optional: If you need more control, you can manually modify the config file
        # This section exports the config, modifies it, and then uploads it separately
        temp_dir = os.path.join(os.getcwd(), "temp_config")
        os.makedirs(temp_dir, exist_ok=True)
        config_path = os.path.join(temp_dir, "config.json")
        # Save the config to a file
        model.config.to_json_file(config_path)
        # Read the config file, modify it, and write it back
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        # Ensure model_type is set in the config
        if 'model_type' not in config_dict or not config_dict['model_type']:
            config_dict['model_type'] = args.model_type
            print(f"Added model_type '{args.model_type}' to config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        # Push model to hub
        model.push_to_hub(args.repo_id, token=args.hf_token)
        tokenizer.push_to_hub(args.repo_id, token=args.hf_token)
        # Upload the modified config.json
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update config.json with model_type",
        )
        # Update README.md after model is pushed
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update model card with adapter information",
        )
        print(f"Model successfully uploaded to {args.repo_id}")
        if args.upload_adapter:
            print(f"Adapter successfully uploaded to {args.adapter_repo_id}")
    except Exception as e:
        print(f"Error pushing model to hub: {e}")
    # Clean up
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
if __name__ == "__main__":
    main()