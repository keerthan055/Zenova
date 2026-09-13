"""Unit tests for dataset integrity checking and checksumming."""
import json
import pytest
from zenova.data.integrity import DatasetIntegrityChecker


def test_compute_and_verify_sha256(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello Zenova Dataset Integrity", encoding="utf-8")

    sha = DatasetIntegrityChecker.compute_sha256(str(f))
    assert len(sha) == 64
    assert DatasetIntegrityChecker.verify_checksum(str(f), sha) is True
    assert DatasetIntegrityChecker.verify_checksum(str(f), "wronghash123") is False


def test_audit_jsonl_valid_and_invalid(tmp_path):
    valid_f = tmp_path / "valid.jsonl"
    with open(valid_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "Hello", "label": "joy"}) + "\n")
        f.write(json.dumps({"text": "World", "label": "sad"}) + "\n")

    is_valid, count, errors = DatasetIntegrityChecker.audit_jsonl(str(valid_f), required_fields=["text", "label"])
    assert is_valid is True
    assert count == 2
    assert len(errors) == 0

    corrupt_f = tmp_path / "corrupt.jsonl"
    with open(corrupt_f, "w", encoding="utf-8") as f:
        f.write("INVALID JSON ROW\n")

    is_valid_c, count_c, errors_c = DatasetIntegrityChecker.audit_jsonl(str(corrupt_f))
    assert is_valid_c is False
    assert len(errors_c) > 0
