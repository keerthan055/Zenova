"""Unit tests for downloader, provenance, and versioning."""
from zenova.data.downloader import LocalCopyDownloader
from zenova.data.provenance import ProvenanceAuditor
from zenova.data.versioning import DatasetLineageTracker, DatasetVersionRecord


def test_local_copy_downloader(tmp_path):
    src_f = tmp_path / "src.json"
    src_f.write_text("{}", encoding="utf-8")

    dl_dir = tmp_path / "dl"
    downloader = LocalCopyDownloader("test_ds", str(src_f), str(dl_dir))
    downloaded = downloader.download()

    assert len(downloaded) == 1
    dest_path = list(downloaded.keys())[0]
    assert (dl_dir / "src.json").exists()


def test_provenance_auditor():
    ok, msg = ProvenanceAuditor.audit_license("MIT")
    assert ok is True
    assert "MIT" in msg

    not_ok, fail_msg = ProvenanceAuditor.audit_license("")
    assert not_ok is False

    cit_ok, _ = ProvenanceAuditor.audit_citation("Long enough citation description for research (2026)")
    assert cit_ok is True

    cit_fail, _ = ProvenanceAuditor.audit_citation("Short")
    assert cit_fail is False


def test_versioning_and_lineage():
    v1 = {"number_of_samples": 100, "labels": ["a"], "checksums": "h1"}
    v2 = {"number_of_samples": 120, "labels": ["a", "b"], "checksums": "h2"}
    changes = DatasetLineageTracker.compare_versions(v1, v2)
    assert len(changes) == 3
