"""Unit tests for ZENOVA configuration loading."""
import pytest
from zenova.core.config import get_system_config, load_yaml_config


def test_load_system_config():
    cfg = get_system_config("configs/system_config.yaml")
    assert cfg.name == "ZENOVA"
    assert cfg.version == "0.1.0"
    assert cfg.safety.strict_mode is True
    assert "988" in cfg.safety.crisis_resources["hotline"]


def test_missing_config_error():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("configs/non_existent.yaml")
