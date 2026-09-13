"""Standard directory layout and path resolution for datasets in ZENOVA."""
import os
from pathlib import Path
from typing import Dict


class DatasetStorage:
    """Manages raw, interim, and processed dataset directory boundaries."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = Path(base_data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.interim_dir = self.base_dir / "interim"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        self.docs_dir = Path("docs") / "datasets"

    def get_dataset_paths(self, dataset_name: str) -> Dict[str, Path]:
        clean_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        paths = {
            "raw": self.raw_dir / clean_name,
            "interim": self.interim_dir / clean_name,
            "processed": self.processed_dir / clean_name,
            "metadata": self.metadata_dir / f"{clean_name}_metadata.json",
            "docs": self.docs_dir / f"{clean_name}.md",
        }
        for k in ["raw", "interim", "processed", "metadata"]:
            if k == "metadata":
                paths[k].parent.mkdir(parents=True, exist_ok=True)
            else:
                paths[k].mkdir(parents=True, exist_ok=True)
        paths["docs"].parent.mkdir(parents=True, exist_ok=True)
        return paths
