"""Standard download and acquisition interfaces for datasets."""
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from zenova.core.logging import get_logger
from zenova.data.integrity import DatasetIntegrityChecker

logger = get_logger("zenova.data.downloader")


class BaseDatasetDownloader(ABC):
    """Abstract base downloader ensuring license and destination path enforcement."""

    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> Dict[str, str]:
        """Acquire the dataset and return a dict of downloaded file paths."""
        pass


class LocalCopyDownloader(BaseDatasetDownloader):
    """Acquires datasets from a verified local source file or directory."""

    def __init__(self, dataset_name: str, source_path: str, target_dir: str):
        super().__init__(dataset_name, target_dir)
        self.source_path = Path(source_path)

    def download(self) -> Dict[str, str]:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found at {self.source_path}")

        dest_file = self.target_dir / self.source_path.name
        shutil.copyfile(self.source_path, dest_file)
        logger.info(f"Copied {self.source_path} -> {dest_file}")

        checksum = DatasetIntegrityChecker.compute_sha256(str(dest_file))
        return {str(dest_file): checksum}
