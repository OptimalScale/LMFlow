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
from transformers import HfArgumentParser
from peft import LoraConfig

from lmflow.args import (
    DiffuserModelArguments, 
    T2IDatasetArguments, 
    AutoArguments,
)
from lmflow.datasets import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline

def main():
    pipeline_name = "diffuser_tuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
    
    parser = HfArgumentParser((DiffuserModelArguments, T2IDatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    
    
    logging_dir = Path(pipeline_args.output_dir, pipeline_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=pipeline_args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=pipeline_args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process and pipeline_args.overwrite_output_dir and os.path.exists(pipeline_args.output_dir):
        shutil.rmtree(pipeline_args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_args.model_name_or_path, subfolder="text_encoder").to("cuda")
    vae = AutoencoderKL.from_pretrained(model_args.model_name_or_path, subfolder="vae").to("cuda")
    
    # dataset = build_t2i_dataset(data_args, tokenizer, text_encoder, vae)
    kwargs = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae}
    dataset = Dataset(data_args, backend="t2i", **kwargs)
    
    del tokenizer, text_encoder, vae
    torch.cuda.empty_cache()
    gc.collect()
    
    model = None
    if model_args.arch_type == "unet":
        model = UNet2DConditionModel.from_pretrained(model_args.model_name_or_path, subfolder=model_args.arch_type)
    elif model_args.arch_type == "transformer":
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
    
    fintuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    accelerator.init_trackers("text2image-finetune", config={
        "data_args": data_args,
        "model_args": model_args,
        "pipeline_args": pipeline_args,
    })
    
    accelerator.wait_for_everyone()
    fintuner.tune(
        accelerator=accelerator,
        model=model, dataset=dataset
    )

if __name__ == '__main__':
    main()
