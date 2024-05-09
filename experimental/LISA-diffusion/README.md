## How to introduce LISA into Stable Diffusion?

[LISA](https://arxiv.org/abs/2403.17919) is an efficient fine-tuning algorithm akin to LoRA, yet it does not introduce additional parameters and can update the full parameters. Unlike LoRA, LISA has only been evaluated on Large Language Models (LLMs) and has not been verified on diffusion models. To address this gap, we have implemented LISA in Stable Diffusion.

### What is New?

We have introduced a layer-by-layer update mechanism to further reduce memory requirements. This is inspired by [GaLore](https://arxiv.org/abs/2403.03507) and formally implemented as:

```python
scheduler_dict = ...
optimizer_dict = ...

    def optimizer_hook(p):
        if p.grad is None:
            del scheduler_dict[p]
            del optimizer_dict[p]
            return
        else:
            if p not in optimizer_dict:
                optimizer_dict[p] = optimizer_class([{"params":p}],
                                                lr=args.learning_rate,
                                                betas=(args.adam_beta1, args.adam_beta2),
                                                weight_decay=args.adam_weight_decay,
                                                eps=args.adam_epsilon)
                optimizer_dict[p] = accelerator.prepare_optimizer(optimizer_dict[p])
            if p not in scheduler_dict:
                scheduler_dict[p] = get_scheduler(
                                args.lr_scheduler,
                                optimizer=optimizer_dict[p],
                                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                num_training_steps=args.max_train_steps * accelerator.num_processes)
                scheduler_dict[p] = accelerator.prepare_scheduler(scheduler_dict[p])
        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(p, args.max_grad_norm)
        optimizer_dict[p].step()
        optimizer_dict[p].zero_grad(set_to_none=True)
        scheduler_dict[p].step()

for p in unet.parameters():
    if p.requires_grad:
        p.register_post_accumulate_grad_hook(optimizer_hook)
```

Without this strategy, LISA would not be able to achieve significant acceleration and memory savings.

### Get Started

you should install necessary packages, including pytorch, diffusers and so on:

```bash
# python==3.10.14
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
pip install -r requirements.txt
```

Note that you should config accelerater by default:

```bash
accelerate config default
```


The subdirectories can then be accessed to execute latent consistency model, instruct pix2pix and diffusion dpo. For different projects, you can execute the corresponding code.

**(1) Latent Consistency Model**

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./lcm_distill_sd_lisa"

accelerate launch train_lcm_distill_sd_wds_lisa.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=no \
    --resolution=512 \
    --learning_rate=1e-5 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=8 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=2 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634
```

**(2) Instruct Pix2Pix**

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_ID="fusing/instructpix2pix-1000-samples"

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix_lisa.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=16 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=no --use_8bit_adam \
    --seed=42
```

**(3) Diffusion DPO**

```bash
accelerate launch train_diffusion_dpo_lisa.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  \
  --output_dir="diffusion-dpo" \
  --mixed_precision="no" \
  --dataset_name=kashif/pickascore \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --checkpointing_steps=2000 \
  --run_validation --validation_steps=200 \
  --seed="0" \
  --report_to="wandb"
```

### Minimum Implementation Code

We also add minimum implementation code in `single_lisa.py`, you can introduce it in your code as follows:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,random
import numpy as np
import accelerate

from single_lisa import LISADiffusion

model = ... # initialize your model
optimizer_class=... # initialize optimizer class akin to which in diffusers
get_scheduler=... # initialize scheduler class akin to which in diffusers
accelerator=... # config your accelerator

lisa_trainer = LISADiffusion(model)
lisa_trainer.register(optimizer_class=optimizer_class,
                      get_scheduler=get_scheduler,
                      accelerator=...,
                      optim_kwargs=...,
                      sched_kwargs=...)
lisa_trainer.insert_hook(optimizer_class=optimizer_class,
                      get_scheduler=get_scheduler,
                      accelerator=...,
                      optim_kwargs=...,
                      sched_kwargs=...)

model = accelerator.prepare_model(model) # this is not necessarily needed

total_count = 0

for i in range(epochs):
    for image in train_loader:

        if total_count % 6 == 0 and total_count != 0: # you can use other number to replace 6
            lisa_trainer.lisa_recall()
        total_count += 1

        Training... # do not forget to remove optimizer.step() and scheduler.step()
```

### Comparison

We have similarly added the training code without lisa in each subdirectory, so the comparison will be extremely convenient:

```bash
|--diffusion_dpo
    |--train_diffusion_dpo_lisa.py
    |--train_diffusion_dpo.py
|--instruct_pix2pix
    |--train_instruct_pix2pix_lisa.py
    |--train_instruct_pix2pix.py
|--latent_consistency_model
    |--train_lcm_distill_sd_wds_lisa.py
    |--train_lcm_distill_sd_wds_lisa.py
```

### Visualization

**(1) Latent Consistency Model**

LISA:

<div align=left>
<img style="width:96%" src="./docs/lcm_lisa_mountain.png">
</div>

LORA:

<div align=left>
<img style="width:96%" src="./docs/lcm_lora_mountain.png">
</div>

**(2) Instruct Pix2Pix**

LISA:

<div align=left>
<img style="width:96%" src="./docs/instruct_lisa_lake.png">
</div>

LORA:

<div align=left>
<img style="width:96%" src="./docs/instruct_lora_lake.png">
</div>