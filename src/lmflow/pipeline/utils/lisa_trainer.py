from typing import Optional

import numpy as np
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback


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

        # Determine the way to access layers based on the model type
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'HymbaForCausalLM': 'model.model.layers',
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            assert lisa_layers_attribute is not None, (
                "Please provide the attribute to access the layers of the model."
            )
            self.layers_attribute = lisa_layers_attribute
        self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers

        self.active_layers_indices = []

    def freeze_all_layers(self):
        layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Check if it's time to switch active layers, including at step 0
        if state.global_step % self.interval_steps == 0:
            self.switch_active_layers()

    def switch_active_layers(self):
        # First, disable gradients for all layers
        self.freeze_all_layers()

            # Randomly select n_layers to activate
        layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
        self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
        print(f"Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

        # Enable gradients only for the selected layers
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True