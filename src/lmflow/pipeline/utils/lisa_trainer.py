import logging
from typing import Optional

import numpy as np
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

# Mapping from model class name to the dot-separated path of transformer layers.
# Add new entries here as new model families are released.
CLASS_TO_LAYERS_MAP = {
    # LLaMA family (1, 2, 3, 3.1, 3.2, 3.3)
    "LlamaForCausalLM": "model.model.layers",
    # Qwen family
    "Qwen2ForCausalLM": "model.model.layers",
    "Qwen2MoeForCausalLM": "model.model.layers",
    # Mistral / Mixtral
    "MistralForCausalLM": "model.model.layers",
    "MixtralForCausalLM": "model.model.layers",
    # Gemma family
    "GemmaForCausalLM": "model.model.layers",
    "Gemma2ForCausalLM": "model.model.layers",
    "Gemma3ForCausalLM": "model.model.layers",
    # Phi family (Microsoft)
    "Phi3ForCausalLM": "model.model.layers",
    "PhiForCausalLM": "model.model.layers",
    # DeepSeek
    "DeepseekV2ForCausalLM": "model.model.layers",
    "DeepseekV3ForCausalLM": "model.model.layers",
    # Cohere (Command R)
    "CohereForCausalLM": "model.model.layers",
    # OLMo (Allen AI)
    "OlmoForCausalLM": "model.model.layers",
    "Olmo2ForCausalLM": "model.model.layers",
    # Falcon
    "FalconForCausalLM": "model.transformer.h",
    # GPT-2
    "GPT2LMHeadModel": "model.transformer.h",
    # GPT-NeoX / Pythia
    "GPTNeoXForCausalLM": "model.gpt_neox.layers",
    # Hymba
    "HymbaForCausalLM": "model.model.layers",
}

# Common layer paths tried in order during dynamic fallback.
_FALLBACK_LAYER_PATHS = [
    "model.model.layers",
    "model.transformer.h",
    "model.gpt_neox.layers",
    "model.layers",
]


def _resolve_layers(model: PreTrainedModel, layers_attribute: str):
    """Walk the dot-separated layers_attribute path on model and return the layer list."""
    obj = model
    for attr in layers_attribute.split(".")[1:]:  # skip leading "model"
        obj = getattr(obj, attr)
    return obj


def _get_layers_attribute(model: PreTrainedModel, lisa_layers_attribute: Optional[str] = None) -> str:
    """Resolve the dot-separated path to the model's transformer layers.

    Resolution order:
    1. User-supplied lisa_layers_attribute override (highest priority).
    2. CLASS_TO_LAYERS_MAP lookup by model class name.
    3. Dynamic introspection across known common paths.

    Raises ValueError if no path can be found.
    """
    unwrapped = model.module if hasattr(model, "module") else model
    model_class_name = type(unwrapped).__name__

    # 1. User override takes highest priority
    if lisa_layers_attribute is not None:
        return lisa_layers_attribute

    # 2. Known architecture map
    if model_class_name in CLASS_TO_LAYERS_MAP:
        return CLASS_TO_LAYERS_MAP[model_class_name]

    # 3. Dynamic fallback — inspect the actual model object
    for path in _FALLBACK_LAYER_PATHS:
        try:
            obj = unwrapped
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (list, nn.ModuleList)):
                logger.warning(
                    "Model class '%s' not in CLASS_TO_LAYERS_MAP. "
                    "Dynamically detected layers at '%s'. "
                    "Consider adding '%s' to CLASS_TO_LAYERS_MAP in lisa_trainer.py.",
                    model_class_name, path, model_class_name,
                )
                return path
        except AttributeError:
            continue

    raise ValueError(
        f"Cannot locate transformer layers for model class '{model_class_name}'. "
        f"Set lisa_layers_attribute in FinetunerArguments to the dot-separated "
        f"path (e.g. 'model.model.layers'), or add '{model_class_name}' to "
        f"CLASS_TO_LAYERS_MAP in src/lmflow/pipeline/utils/lisa_trainer.py."
    )


class DynamicLayerActivationCallback(TrainerCallback):
    def __init__(
        self,
        n_layers: int,
        interval_steps: int,
        model: PreTrainedModel,
        lisa_layers_attribute: Optional[str] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.interval_steps = interval_steps
        self.model = model

        self.layers_attribute = _get_layers_attribute(model, lisa_layers_attribute)
        self.total_layers = len(_resolve_layers(self.model, self.layers_attribute))

        self.active_layers_indices = []

    def freeze_all_layers(self):
        layers = _resolve_layers(self.model, self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.interval_steps == 0:
            self.switch_active_layers()

    def switch_active_layers(self):
        self.freeze_all_layers()

        layers = _resolve_layers(self.model, self.layers_attribute)
        self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
        logger.info("Activating layers at indices: %s for the next steps.", self.active_layers_indices)

        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True
