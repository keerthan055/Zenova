"""Dataset integrity verification, checksumming, and corruption audits."""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class DatasetIntegrityChecker:
    """Calculates cryptographic hashes and validates file structural integrity."""

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def verify_checksum(cls, file_path: str, expected_hash: str) -> bool:
        actual_hash = cls.compute_sha256(file_path)
        return actual_hash.lower() == expected_hash.lower()

    @staticmethod
    def audit_jsonl(file_path: str, required_fields: Optional[List[str]] = None) -> Tuple[bool, int, List[str]]:
        """Audit a JSON Lines file for corruption, empty lines, and missing keys."""
        p = Path(file_path)
        if not p.exists():
            return False, 0, [f"File {file_path} does not exist."]

        valid_count = 0
        errors = []
        required = required_fields or []

        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean = line.strip()
                if not clean:
                    errors.append(f"Line {line_idx}: Empty row detected.")
                    continue
                try:
                    record = json.loads(clean)
                    missing = [k for k in required if k not in record or record[k] is None]
                    if missing:
                        errors.append(f"Line {line_idx}: Missing required keys {missing}")
                    else:
                        valid_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_idx}: Invalid JSON syntax - {str(e)}")

        is_valid = len(errors) == 0 and valid_count > 0
        return is_valid, valid_count, errors
