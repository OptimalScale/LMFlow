import types
import pytest
import torch.nn as nn

from lmflow.pipeline.utils.lisa_trainer import (
    CLASS_TO_LAYERS_MAP,
    _get_layers_attribute,
)


def make_mock_model(class_name: str, layers_path: str = "model.model.layers", num_layers: int = 4):
    """Build a minimal mock model with layers at the given dot-separated path.

    Uses SimpleNamespace for nested attributes so we avoid nn.Module.__init__
    complexity. The top-level object is given the requested class name so that
    type(model).__name__ returns it correctly.

    For example, layers_path="model.model.layers" creates:
        mock.model.model.layers = ModuleList([...])
    """
    layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_layers)])

    current = layers
    for part in reversed(layers_path.split(".")):
        parent = types.SimpleNamespace()
        setattr(parent, part, current)
        current = parent

    MockClass = type(class_name, (object,), {})
    instance = object.__new__(MockClass)
    instance.__dict__.update(vars(current))
    return instance


class TestGetLayersAttribute:

    def test_known_architecture_uses_map(self):
        """LLaMA is in CLASS_TO_LAYERS_MAP — should return the mapped path directly."""
        model = make_mock_model("LlamaForCausalLM", "model.model.layers")
        result = _get_layers_attribute(model)
        assert result == "model.model.layers"

    def test_newly_added_architecture_gemma2(self):
        """Gemma2 was added to the expanded map — should resolve without fallback."""
        model = make_mock_model("Gemma2ForCausalLM", "model.model.layers")
        result = _get_layers_attribute(model)
        assert result == "model.model.layers"
        assert "Gemma2ForCausalLM" in CLASS_TO_LAYERS_MAP

    def test_falcon_maps_to_transformer_h(self):
        """FalconForCausalLM maps to model.transformer.h — verifies non-default path entries."""
        model = make_mock_model("FalconForCausalLM", "model.transformer.h")
        result = _get_layers_attribute(model)
        assert result == "model.transformer.h"

    def test_user_override_takes_precedence_over_map(self):
        """User-supplied lisa_layers_attribute must win even for known architectures.

        Uses a custom path that differs from both the map entry and all fallback
        paths, so the only way the test passes is if the override is truly used.
        """
        model = make_mock_model("LlamaForCausalLM", "model.model.layers")
        result = _get_layers_attribute(model, lisa_layers_attribute="model.custom.blocks")
        assert result == "model.custom.blocks"

    def test_dynamic_fallback_finds_transformer_h(self):
        """Unknown model with layers at model.transformer.h — fallback iterates past first entry."""
        model = make_mock_model("BrandNewGPTModel", "model.transformer.h")
        result = _get_layers_attribute(model)
        assert result == "model.transformer.h"

    def test_completely_unknown_model_raises_valueerror(self):
        """Unknown model with no recognizable layer path should raise a clear ValueError."""
        model = make_mock_model("WeirdModelWithNoLayers", "model.model.layers")
        model.__dict__.clear()
        with pytest.raises(ValueError, match="Cannot locate transformer layers"):
            _get_layers_attribute(model)

    def test_dataparallel_wrapped_model_unwrapped(self):
        """Model wrapped in DataParallel (.module) should be unwrapped before class lookup."""
        inner = make_mock_model("LlamaForCausalLM", "model.model.layers")

        # Simulate DataParallel wrapping: outer object has a .module attribute
        WrapperClass = type("DataParallel", (object,), {})
        wrapper = object.__new__(WrapperClass)
        wrapper.module = inner

        result = _get_layers_attribute(wrapper)
        assert result == "model.model.layers"
