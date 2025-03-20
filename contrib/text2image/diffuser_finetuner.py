import json
import os
import logging
import gc
import copy

from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from lmflow.pipeline.finetuner import BaseTuner

from diffuser_args import T2IDatasetArguments, DiffuserModelArguments, DiffuserTunerArguments

logger = logging.getLogger(__name__)

def log_validation(
    pipeline,
    accelerator: Accelerator,
    pipeline_args: dict,
    save_dir,
    global_step,
):
    pipeline.to(accelerator.device)
    pipeline.vae.to(torch.float32)
    
    with torch.no_grad():
        prompt_images = [
            (pipeline_arg["prompt"], pipeline(**pipeline_arg).images[0]) for pipeline_arg in pipeline_args
        ]
        
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {prompt}") for i, (prompt, image) in enumerate(prompt_images)
                    ]
                }
            )
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (prompt, image) in enumerate(prompt_images):
        image.save(os.path.join(save_dir, f"{prompt.replace(' ', '_')}.png"))
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    return

class DiffuserModelTuner(BaseTuner):
    """Initializes the `RewardModelTuner` class.

    Parameters
    ----------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : RewardModelTunerArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.
    """
    def __init__(
        self, 
        model_args: DiffuserModelArguments,
        data_args: T2IDatasetArguments,
        finetuner_args: DiffuserTunerArguments,
        *args, 
        **kwargs
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.finetuner_args = finetuner_args
    
    def tune(
        self,
        accelerator: Accelerator,
        model,
        dataset,
    ):
        dataloader = DataLoader(dataset=dataset, batch_size=self.finetuner_args.train_batch_size, shuffle=True)
        
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_args.model_name_or_path, subfolder="scheduler")
        
        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model    
        
        # filter trainable parameters
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
        accelerator.print(len(params_to_optimize))
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.finetuner_args.learning_rate,
            weight_decay=self.finetuner_args.weight_decay
        )
        
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
        )
        
        model, dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, dataloader, optimizer, lr_scheduler
        )
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            
        progress_bar = tqdm(
            range(self.finetuner_args.num_train_epochs * len(dataloader)),
            desc="Training",
            disable=not accelerator.is_main_process
        )
        
        global_step = 0
        for epoch in range(self.finetuner_args.num_train_epochs):
            model.train()
            for batch in dataloader:
                clean_latents = batch["image"].to(dtype=weight_dtype)
                text_embedding = batch["text"].to(dtype=weight_dtype)
                
                bsz, channel, height, width = clean_latents.shape
                noise = torch.randn_like(clean_latents).to(dtype=weight_dtype)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_latents.device
                )
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
                model_pred = model(
                    noisy_latents, timesteps, text_embedding,
                )[0]
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
            
                if accelerator.is_main_process:
                    logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    global_step += 1
                
                # validation
                if accelerator.is_main_process and self.finetuner_args.do_valid and self.data_args.validation_file is not None:
                    if global_step % self.finetuner_args.valid_steps == 0:
                        with torch.no_grad():
                            pipeline = DiffusionPipeline.from_pretrained(
                                self.model_args.model_name_or_path,
                                torch_dtype=weight_dtype,
                            )
                            if self.model_args.model_type == "unet":
                                pipeline.unet = unwrap_model(model)
                            elif self.model_args.model_type == "transformer":
                                pipeline.transformer = unwrap_model(model)
                            else:
                                raise ValueError(f"Unknown model type {self.model_args.model_type}")
                            
                            with open(os.path.join(self.data_args.dataset_path, self.data_args.validation_file), "r") as f:
                                validation_data = json.load(f)
                            generator = torch.Generator(device=accelerator.device).manual_seed(self.finetuner_args.valid_seed)
                            pipeline_args = [
                                {"prompt": item["text"], "generator": generator, "width": self.data_args.image_size, "height": self.data_args.image_size}
                                for item in validation_data["instances"]
                            ]
                            log_validation(
                                pipeline,
                                accelerator,
                                pipeline_args,
                                os.path.join(self.finetuner_args.output_dir, f"step_{global_step}_validation"),
                                global_step,
                            )
                                
                if accelerator.is_main_process and global_step % self.finetuner_args.save_steps == 0:
                    os.makedirs(os.path.join(self.finetuner_args.output_dir, f"checkpoints"), exist_ok=True)
                    if len(os.listdir(os.path.join(self.finetuner_args.output_dir, f"checkpoints"))) > self.finetuner_args.max_checkpoints:
                        os.remove(os.path.join(self.finetuner_args.output_dir, f"checkpoints", sorted(os.listdir(os.path.join(self.finetuner_args.output_dir, f"checkpoints")))[0]))
                    if self.model_args.use_lora:
                        temp_model = unwrap_model(copy.deepcopy(model))
                        temp_model = temp_model.to(torch.float32)
                        model_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(temp_model))
                        LoraLoaderMixin.save_lora_weights(
                            save_directory=os.path.join(self.finetuner_args.output_dir, f"checkpoints", f"final"),
                            unet_lora_layers=model_lora_state_dict if self.model_args.model_type == "unet" else None,
                            transformer_lora_layers=model_lora_state_dict if self.model_args.model_type == "transformer" else None,
                        )
                        del temp_model
                    else:
                        accelerator.save(
                            accelerator.get_state_dict(model),
                            os.path.join(self.finetuner_args.output_dir, f"checkpoints", f"final.pt")
                        )
        
        accelerator.wait_for_everyone()
        progress_bar.close()
        if accelerator.is_main_process:
            if self.finetuner_args.do_test and self.data_args.test_file is not None:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.model_args.model_name_or_path,
                    torch_dtype=weight_dtype,
                )
                if self.model_args.model_type == "unet":
                    pipeline.unet = unwrap_model(model)
                elif self.model_args.model_type == "transformer":
                    pipeline.transformer = unwrap_model(model)
                else:
                    raise ValueError(f"Unknown model type {self.model_args.model_type}")
                
                with open(os.path.join(self.data_args.dataset_path, self.data_args.test_file), "r") as f:
                    test_data = json.load(f)
                generator = torch.Generator(device=accelerator.device).manual_seed(self.finetuner_args.test_seed)
                pipeline_args = [
                    {"prompt": item["text"], "generator": generator, "width": self.data_args.image_size, "height": self.data_args.image_size}
                    for item in test_data["instances"]
                ]
                log_validation(
                    pipeline,
                    accelerator,
                    pipeline_args,
                    os.path.join(self.finetuner_args.output_dir, f"test_final"),
                    global_step,
                )
            
            os.makedirs(os.path.join(self.finetuner_args.output_dir, f"checkpoints"), exist_ok=True)
            if self.model_args.use_lora:
                model = unwrap_model(model)
                model = model.to(torch.float32)
                model_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                LoraLoaderMixin.save_lora_weights(
                    save_directory=os.path.join(self.finetuner_args.output_dir, f"checkpoints", f"final"),
                    unet_lora_layers=model_lora_state_dict if self.model_args.model_type == "unet" else None,
                    transformer_lora_layers=model_lora_state_dict if self.model_args.model_type == "transformer" else None,
                )
                # pipeline.load_lora_weights(output_dir, weight_name="pytorch_lora_weights.safetensors")
            else:
                accelerator.save(
                    accelerator.get_state_dict(model),
                    os.path.join(self.finetuner_args.output_dir, f"checkpoints", f"final.pt")
                )
                
        return
    