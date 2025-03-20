#!/usr/bin/env python
# coding=utf-8

"""This Python code defines a class T2I Dataset.
"""
import json
from PIL import Image
import os.path as osp
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from diffuser_args import T2IDatasetArguments

logger = logging.getLogger(__name__)

class CustomT2IDataset(Dataset):
    """Dataset for T2I data"""
    
    def __init__(self, data_args: T2IDatasetArguments):
        self.data_args = data_args
        self.image_folder = osp.join(data_args.dataset_path, data_args.image_folder)
        self.data_file = osp.join(data_args.dataset_path, data_args.train_file)
        
        self.data_dict = json.load(open(self.data_file, "r"))
        assert self.data_dict["type"] == "text-image", "The dataset type must be text-image."
        
        self.data_instances = self.data_dict["instances"]
    
    def __len__(self):
        return len(self.data_instances)
    
    def __getitem__(self, idx):
        instance = self.data_instances[idx]
        image_path = osp.join(self.image_folder, instance["image"])
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        return {
            "image": image,
            "text": instance["text"],
        }

class EncodePreprocessor(object):
    def __init__(self, data_args: T2IDatasetArguments, kind: str = "simple", **kwargs):
        self.transform = transforms.Compose(
            [
                transforms.Resize(data_args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(data_args.image_size) if data_args.image_crop_type == "center" else transforms.RandomCrop(data_args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        
        self.pre_func = None
        if kind == "simple":
            self.register_simple_func(**kwargs)
    
    def register_simple_func(self, tokenizer, text_encoder, vae):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        
        def simple_func(data_item):
            image = self.transform(data_item["image"])
            latents = self.vae.encode(image.to(self.vae.device, dtype=self.vae.dtype).unsqueeze(0)).latent_dist.sample()
            encoded_image = latents * self.vae.config.scaling_factor
            encoded_image = encoded_image.detach()
            encoded_image=encoded_image.squeeze(0).cpu()
            
            max_length = self.tokenizer.model_max_length
            tokens = self.tokenizer([data_item["text"]], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            encoded_text = self.text_encoder(tokens.to(self.text_encoder.device))[0]
            encoded_text = encoded_text.detach()
            encoded_text =encoded_text.squeeze(0).cpu()
            
            return {
                "image": encoded_image,
                "text": encoded_text,
            }
            
        self.pre_func = simple_func
        
    def __call__(self, data_item):
        return self.pre_func(data_item)     
   
class PreprocessedT2IDataset(Dataset):
    "Preprocess dataset with prompt"
    
    def __init__(self, raw_dataset:Dataset, data_args: T2IDatasetArguments, preprocessor:EncodePreprocessor):
        self.data_dict = []
        
        logger.info("Preprocessing data ...")
        for data_item in tqdm(raw_dataset):
            self.data_dict.append(preprocessor(data_item))
            
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        return self.data_dict[idx]

def build_t2i_dataset(data_args: T2IDatasetArguments, tokenizer, text_encoder, vae):
    raw_dataset = CustomT2IDataset(data_args)
    # dataset = SimpleT2IDataset(raw_dataset, data_args, tokenizer, text_encoder, vae)
    preprocessor = EncodePreprocessor(kind="simple", data_args=data_args, tokenizer=tokenizer, text_encoder=text_encoder, vae=vae)
    dataset = PreprocessedT2IDataset(raw_dataset, data_args, preprocessor)
    
    return dataset