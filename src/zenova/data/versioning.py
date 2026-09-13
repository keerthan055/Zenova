"""Dataset version tracking and lineage management."""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class DatasetVersionRecord(BaseModel):
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str
    checksum: str
    changes: List[str] = Field(default_factory=list)


class DatasetLineageTracker:
    """Tracks version transitions and transformations for datasets."""

    @staticmethod
    def compare_versions(v1_meta: Dict[str, Any], v2_meta: Dict[str, Any]) -> List[str]:
        changes = []
        if v1_meta.get("number_of_samples") != v2_meta.get("number_of_samples"):
            changes.append(f"Sample count changed: {v1_meta.get('number_of_samples')} -> {v2_meta.get('number_of_samples')}")
        if v1_meta.get("labels") != v2_meta.get("labels"):
            changes.append(f"Labels modified: {len(v1_meta.get('labels', []))} -> {len(v2_meta.get('labels', []))}")
        if v1_meta.get("checksums") != v2_meta.get("checksums"):
            changes.append("File checksums updated")
        return changes
