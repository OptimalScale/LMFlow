import glob
import torch
from transformers import LlamaConfig
from tqdm import tqdm


def update_custom_config(config, model_args):
    if model_args.llm_model_name_or_path is not None:
        text_config = LlamaConfig.from_pretrained(
            model_args.llm_model_name_or_path)
        config.text_config = text_config
    config.with_qformer = model_args.with_qformer
    config.custom_vision_model = model_args.custom_vision_model
    if model_args.custom_vision_model:
        # config.vision_model_args = model_args
        config.image_encoder_name_or_path = \
            model_args.image_encoder_name_or_path
        config.vision_select_layer = model_args.vision_select_layer
        if getattr(model_args, "vision_select_feature", None) is not None:
            config.vision_select_feature = model_args.vision_select_feature
    return config


def load_llava_pretrain_model(model, checkpoint_path):
    checkpoint_path = glob.glob(checkpoint_path)
    for path in tqdm(checkpoint_path):
        state_dict = torch.load(path, map_location="cpu")
        new_state_dict = adapt_llava_model_to_lmflow_type(state_dict)
        # modify the name of the key
        # import pdb; pdb.set_trace()
        lmflow_keys = model.state_dict().keys()
        for key in new_state_dict.keys():
            if key not in lmflow_keys:
                print("key not in lmflow_keys: ", key)
        model.load_state_dict(new_state_dict, strict=False)
    return model

def adapt_llava_model_to_lmflow_type(state_dict):
    new_state_dict = {}
    for key, item in state_dict.items():
        key = key.replace("model.layers", "language_model.model.layers")
        key = key.replace("model.embed_tokens",
                          "language_model.model.embed_tokens")
        key = key.replace("model.mm_projector", "language_projection")
        key = key.replace("lm_head", "language_model.lm_head")
        key = key.replace("model.norm", "language_model.model.norm")
        if "vision_tower" in key:
            continue
        new_state_dict[key] = item
    return new_state_dict