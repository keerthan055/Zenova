"""Unit tests for model versioning, manifest loading, and artifact verification."""
import os
import json
import tempfile
from pathlib import Path
from zenova.models.versioning import ModelVersionManager, get_model_version_manager


def test_model_version_manager_loads_manifest():
    """Verify ModelVersionManager loads the canonical manifest."""
    manager = get_model_version_manager()
    assert manager is not None
    assert manager.manifest is not None
    assert "models" in manager.manifest

    # Check emotion model metadata
    emo_info = manager.get_model_info("emotion")
    assert emo_info is not None
    assert emo_info["version"] == "1.0.0"
    assert emo_info["task"] == "multi_class_classification"

    # Check risk model metadata
    risk_info = manager.get_model_info("risk")
    assert risk_info is not None
    assert risk_info["baseline_sensitivity"] == 1.0


def test_model_version_query():
    """Verify version resolution with default fallback."""
    manager = get_model_version_manager()
    assert manager.get_version("emotion") == "1.0.0"
    assert manager.get_version("non_existent_model", default="2.0.0") == "2.0.0"


def test_model_artifact_verification():
    """Verify artifact presence and SHA-256 computation."""
    manager = get_model_version_manager()
    report = manager.verify_model_artifacts()

    assert "total_models" in report
    assert report["total_models"] >= 4
    assert "models" in report
    assert "emotion" in report["models"]
    assert "symptoms" in report["models"]
    assert "risk" in report["models"]
    assert "strategy" in report["models"]


def test_compute_sha256_missing_file():
    """Verify compute_sha256 returns None for missing files."""
    manager = get_model_version_manager()
    assert manager.compute_sha256("non_existent_file.pt") is None
