"""Dataset catalog discovery and validation API endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from zenova.data.registry import DatasetRegistry
from zenova.data.integrity import DatasetIntegrityChecker

router = APIRouter(prefix="/api/v1/datasets", tags=["Dataset Management"])
registry = DatasetRegistry()


class ValidateDatasetRequest(BaseModel):
    dataset_name: str


@router.get("")
async def list_datasets():
    """List all registered research datasets and their metadata."""
    return {
        "count": len(registry.list_datasets()),
        "datasets": registry.list_datasets()
    }


@router.get("/{name}")
async def get_dataset(name: str):
    """Retrieve detailed metadata, splits, and statistics for a specific dataset."""
    ds = registry.get_dataset(name)
    if not ds:
        # Try case-insensitive matching
        for k, v in registry._catalog.items():
            if k.lower() == name.lower() or k.lower().replace(" ", "_") == name.lower():
                ds = v
                break
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found in registry.")
    return ds.model_dump(mode="json")


@router.post("/validate")
async def validate_dataset(req: ValidateDatasetRequest):
    """Audit a registered dataset's file checksums and schema integrity."""
    ds = registry.get_dataset(req.dataset_name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_name}' not found in registry.")

    results = {}
    for split, hash_val in ds.checksums.items():
        if split in ["train", "val", "test"]:
            file_p = f"{ds.local_storage_location['processed']}/{split}.jsonl"
            try:
                is_valid = DatasetIntegrityChecker.verify_checksum(file_p, hash_val)
                audit_ok, count, errors = DatasetIntegrityChecker.audit_jsonl(file_p)
                results[split] = {
                    "checksum_valid": is_valid,
                    "records_valid": audit_ok,
                    "sample_count": count,
                    "errors": errors[:3]
                }
            except Exception as e:
                results[split] = {"error": str(e)}

    return {
        "dataset_name": ds.dataset_name,
        "integrity_audit": results,
        "leakage_check_passed": ds.leakage_check_passed
    }
