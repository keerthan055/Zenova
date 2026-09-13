"""Dataset metadata catalog and discovery registry for ZENOVA."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from zenova.data.metadata import DatasetMetadata
from zenova.core.logging import get_logger

logger = get_logger("zenova.data.registry")


class DatasetRegistry:
    """Catalog of verified research datasets used in ZENOVA."""

    def __init__(self, metadata_dir: str = "data/metadata"):
        self.metadata_dir = Path(metadata_dir)
        self._catalog: Dict[str, DatasetMetadata] = {}
        self.load_all_metadata()

    def load_all_metadata(self) -> None:
        if not self.metadata_dir.exists():
            return
        for file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Filter out legacy keys if necessary or cast
                ds = DatasetMetadata(**data)
                self._catalog[ds.dataset_name] = ds
                logger.info(f"Loaded dataset from registry: {ds.dataset_name} (Task: {ds.intended_task})")
            except Exception as e:
                logger.debug(f"Could not load {file} into new DatasetMetadata: {e}")

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        return self._catalog.get(name)

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [ds.model_dump(mode="json") for ds in self._catalog.values()]

    def register_dataset(self, metadata: DatasetMetadata) -> None:
        self._catalog[metadata.dataset_name] = metadata
        clean_name = metadata.dataset_name.lower().replace(" ", "_").replace("-", "_")
        out_p = self.metadata_dir / f"{clean_name}_metadata.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2)
        logger.info(f"Registered and saved dataset '{metadata.dataset_name}' to {out_p}")
