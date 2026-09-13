"""Comprehensive dataset metadata schema for research provenance and tracking."""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class DatasetSplitInfo(BaseModel):
    train_count: int = Field(default=0, ge=0)
    val_count: int = Field(default=0, ge=0)
    test_count: int = Field(default=0, ge=0)
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    random_seed: int = 42
    split_strategy: str = "stratified"


class DatasetMetadata(BaseModel):
    """Standard schema conforming to research and governance requirements."""
    dataset_name: str = Field(..., description="Canonical human-readable dataset name")
    version: str = Field(default="1.0.0", description="Semantic version string (e.g. 1.0.0)")
    source: str = Field(..., description="Originating organization, research group, or university")
    official_url: str = Field(..., description="Canonical repository, paper URL, or download portal")
    license: str = Field(..., description="Permitted license (e.g. Apache-2.0, MIT, CC-BY-4.0)")
    citation: str = Field(..., description="Full academic reference citation in APA / ACL format")
    download_date: str = Field(..., description="ISO 8601 date string of dataset acquisition")
    local_storage_location: Dict[str, str] = Field(
        ...,
        description="Resolved local filesystem locations: raw, interim, processed, metadata"
    )
    intended_task: str = Field(..., description="Specific ML task (e.g. support_strategy_planning, emotion_analysis)")
    features: List[str] = Field(default_factory=list, description="Input feature names or textual modalities")
    labels: List[str] = Field(default_factory=list, description="Target label taxonomy or classification classes")
    number_of_samples: Dict[str, int] = Field(
        default_factory=lambda: {"total": 0, "raw": 0, "processed": 0},
        description="Counts of samples across stages"
    )
    train_validation_test_split: DatasetSplitInfo = Field(default_factory=DatasetSplitInfo)
    preprocessing_steps: List[str] = Field(default_factory=list, description="Ordered list of applied transformations")
    checksums: Dict[str, str] = Field(default_factory=dict, description="SHA-256 hashes of files")
    leakage_check_passed: bool = Field(default=False, description="Flag indicating strict cross-split leakage verification")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics: distribution, lengths, missingness")
    explicit_non_goals: List[str] = Field(default_factory=list, description="Prohibited or unsupported uses")
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Version must follow semantic versioning (MAJOR.MINOR.PATCH), got: {v}")
        return v
