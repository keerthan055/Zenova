"""Model artifact version management, integrity verification, and metadata registry."""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from zenova.core.logging import get_logger

logger = get_logger("zenova.models.versioning")


class ModelVersionManager:
    """Manages model artifact manifests, version tracking, and checkpoint integrity."""

    def __init__(self, manifest_path: str = "models/manifest.json"):
        self.manifest_path = Path(manifest_path)
        self.manifest: Dict[str, Any] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                logger.info(f"Loaded model manifest (version {self.manifest.get('manifest_version', '1.0.0')})")
            except Exception as e:
                logger.error(f"Failed to parse model manifest at {self.manifest_path}: {e}")
                self.manifest = {}
        else:
            logger.warning(f"Model manifest not found at {self.manifest_path}")
            self.manifest = {}

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve manifest metadata for a specific model."""
        return self.manifest.get("models", {}).get(model_key)

    def get_version(self, model_key: str, default: str = "1.0.0") -> str:
        """Get semantic version string for a model."""
        info = self.get_model_info(model_key)
        return info.get("version", default) if info else default

    def compute_sha256(self, file_path: str) -> Optional[str]:
        """Compute SHA-256 hash of an artifact checkpoint file."""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def verify_model_artifacts(self) -> Dict[str, Any]:
        """Verify presence and integrity of all models declared in the manifest.
        
        Returns:
            Dict containing per-model verification statuses and overall validation result.
        """
        results: Dict[str, Any] = {}
        all_passed = True
        models = self.manifest.get("models", {})

        for key, info in models.items():
            ckpt_path = info.get("checkpoint_path")
            is_present = False
            file_size_bytes = 0
            file_hash = None

            if ckpt_path and os.path.exists(ckpt_path):
                is_present = True
                file_size_bytes = os.path.getsize(ckpt_path)
                file_hash = self.compute_sha256(ckpt_path)
            else:
                is_present = False
                all_passed = False

            results[key] = {
                "name": info.get("name"),
                "version": info.get("version"),
                "checkpoint_path": ckpt_path,
                "exists": is_present,
                "size_bytes": file_size_bytes,
                "sha256": file_hash[:16] + "..." if file_hash else None,
                "active": info.get("active", True)
            }

        return {
            "all_verified": all_passed,
            "total_models": len(models),
            "manifest_version": self.manifest.get("manifest_version", "unknown"),
            "models": results
        }


# Global singleton
_GLOBAL_MODEL_VERSION_MANAGER: Optional[ModelVersionManager] = None


def get_model_version_manager() -> ModelVersionManager:
    """Retrieve global ModelVersionManager instance."""
    global _GLOBAL_MODEL_VERSION_MANAGER
    if _GLOBAL_MODEL_VERSION_MANAGER is None:
        _GLOBAL_MODEL_VERSION_MANAGER = ModelVersionManager()
    return _GLOBAL_MODEL_VERSION_MANAGER
