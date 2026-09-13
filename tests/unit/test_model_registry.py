"""Unit tests for ModelRegistry and dynamic swapping."""
import pytest
from zenova.models.registry import ModelRegistry
from zenova.emotion.analyzer import EmotionTransformerAnalyzer


def test_model_registry_resolution():
    registry = ModelRegistry()
    assert registry.get_active_provider("emotion") == "transformer"
    instance = registry.get_module_instance("emotion")
    assert isinstance(instance, EmotionTransformerAnalyzer)


def test_model_registry_switch_provider():
    registry = ModelRegistry()
    # Can switch back to placeholder
    registry.set_active_provider("emotion", "placeholder")
    assert registry.get_active_provider("emotion") == "placeholder"

    # Switch back to transformer
    registry.set_active_provider("emotion", "transformer")
    assert registry.get_active_provider("emotion") == "transformer"

    # Attempting to switch to an unregistered provider raises error
    with pytest.raises(Exception):
        registry.set_active_provider("emotion", "non_existent_provider")
