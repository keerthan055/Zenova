"""Unit tests for DatasetMetadata schema and versioning."""
import pytest
from pydantic import ValidationError
from zenova.data.metadata import DatasetMetadata, DatasetSplitInfo


def test_dataset_metadata_valid():
    meta = DatasetMetadata(
        dataset_name="Test Dataset",
        version="1.0.0",
        source="Zenova Research",
        official_url="https://zenova.test/ds",
        license="MIT",
        citation="Zenova (2026)",
        download_date="2026-09-09",
        local_storage_location={"raw": "data/raw/test"},
        intended_task="emotion_analysis",
        features=["text"],
        labels=["happy", "sad"]
    )
    assert meta.dataset_name == "Test Dataset"
    assert meta.version == "1.0.0"


def test_dataset_metadata_invalid_semver():
    with pytest.raises(ValidationError):
        DatasetMetadata(
            dataset_name="Test Dataset",
            version="1.0",  # Invalid: requires 3 parts
            source="Zenova Research",
            official_url="https://zenova.test/ds",
            license="MIT",
            citation="Zenova (2026)",
            download_date="2026-09-09",
            local_storage_location={"raw": "data/raw/test"},
            intended_task="emotion_analysis"
        )
