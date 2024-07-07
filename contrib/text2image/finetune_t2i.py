import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"
import shutil
from pathlib import Path
import gc

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import (
    AutoTokenizer,
    CLIPTextModel
)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffuser_args import T2IDatasetArguments, DiffuserModelArguments, DiffuserTunerArguments
from t2i_dataset import build_t2i_dataset
from diffuser_finetuner import DiffuserModelTuner
from transformers import HfArgumentParser

from peft import LoraConfig

def main():
    parser = HfArgumentParser((DiffuserModelArguments, T2IDatasetArguments, DiffuserTunerArguments))
    model_args, data_args, tuner_args = parser.parse_args_into_dataclasses()
    
    
    logging_dir = Path(tuner_args.output_dir, tuner_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=tuner_args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=tuner_args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process and tuner_args.overwrite_output_dir and os.path.exists(tuner_args.output_dir):
        shutil.rmtree(tuner_args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_args.model_name_or_path, subfolder="text_encoder").to("cuda")
    vae = AutoencoderKL.from_pretrained(model_args.model_name_or_path, subfolder="vae").to("cuda")
    
    dataset = build_t2i_dataset(data_args, tokenizer, text_encoder, vae)
    
    del tokenizer, text_encoder, vae
    torch.cuda.empty_cache()
    gc.collect()
    
    model = None
    if model_args.model_type == "unet":
        model = UNet2DConditionModel.from_pretrained(model_args.model_name_or_path, subfolder=model_args.model_type)
    elif model_args.model_type == "transformer":
        raise NotImplementedError("Transformer model is not implemented.")
    else:
        raise ValueError("The model type is not supported.")
    if model_args.use_lora:
        accelerator.print(f"Using LoRA of {model_args.lora_target_modules} for training")
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=model_args.lora_target_modules,
        )
        model.add_adapter(lora_config)
    else:
        model.requires_grad_(True)
    
    fintuner = DiffuserModelTuner(model_args, data_args, tuner_args)
    accelerator.init_trackers("text2image-finetune", config={
        "data_args": data_args,
        "model_args": model_args,
        "tuner_args": tuner_args,
    })
    
    accelerator.wait_for_everyone()
    fintuner.tune(
        accelerator=accelerator,
        model=model, dataset=dataset
    )

if __name__ == '__main__':
    main()
