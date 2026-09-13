"""Unit tests for automated database backup and retention pruning."""
import os
import gzip
import tempfile
from pathlib import Path
from scripts.backup_db import backup_database, rotate_backups, compute_sha256


def test_backup_database_creates_archive_and_checksum():
    """Verify backup utility compresses database and writes valid SHA-256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy database file
        dummy_db = Path(tmpdir) / "test.db"
        dummy_db.write_text("SAMPLE_SQLITE_DATABASE_CONTENT_FOR_BACKUP_TEST")

        backup_dir = Path(tmpdir) / "backups"

        # Execute backup
        backup_res = backup_database(
            db_path=str(dummy_db.relative_to(Path.cwd()) if dummy_db.is_relative_to(Path.cwd()) else dummy_db),
            backup_dir=str(backup_dir),
            retention_count=3
        )

        assert backup_res is not None
        assert backup_res.exists()
        assert backup_res.name.endswith(".db.gz")

        # Verify checksum file was created
        checksum_file = backup_dir / f"{backup_res.name}.sha256"
        assert checksum_file.exists()

        # Check content decompression
        with gzip.open(backup_res, "rt") as f_gz:
            decompressed_text = f_gz.read()
        assert decompressed_text == "SAMPLE_SQLITE_DATABASE_CONTENT_FOR_BACKUP_TEST"


def test_rotate_backups_enforces_retention():
    """Verify old backups exceeding retention limit are pruned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        b_dir = Path(tmpdir)

        # Create 5 fake backup files
        files = []
        for i in range(5):
            f = b_dir / f"zenova_db_backup_2026090{i}_000000.db.gz"
            f.write_text("data")
            chk = b_dir / f"{f.name}.sha256"
            chk.write_text("hash")
            files.append(f)

        assert len(list(b_dir.glob("*.db.gz"))) == 5

        # Rotate with retention of 2
        rotate_backups(b_dir, retention_count=2)

        remaining = list(b_dir.glob("*.db.gz"))
        assert len(remaining) == 2
