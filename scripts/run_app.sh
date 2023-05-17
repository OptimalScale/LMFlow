#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerator_singlegpu_config.yaml service/app.py 
