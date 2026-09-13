"""Database backup utility with gzip compression, SHA-256 checksums, and retention rotation."""
import os
import sys
import gzip
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add src to pythonpath
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.backup")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def backup_database(
    db_path: str = "data/zenova.db",
    backup_dir: str = "backups",
    retention_count: int = 7
) -> Optional[Path]:
    """Create timestamped, compressed database backup with checksum verification."""
    source_file = root_dir / db_path
    if not source_file.exists():
        logger.warning(f"Database source file not found at {source_file}. Skipping backup.")
        return None

    target_dir = root_dir / backup_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_filename = f"zenova_db_backup_{timestamp}.db.gz"
    backup_path = target_dir / backup_filename
    checksum_path = target_dir / f"{backup_filename}.sha256"

    logger.info(f"Creating compressed backup of {source_file} -> {backup_path}...")
    try:
        with open(source_file, "rb") as f_in:
            with gzip.open(backup_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Compute and save checksum
        checksum = compute_sha256(backup_path)
        with open(checksum_path, "w", encoding="utf-8") as f_chk:
            f_chk.write(f"{checksum}  {backup_filename}\n")

        size_kb = backup_path.stat().st_size / 1024
        logger.info(f"Backup created successfully: {backup_filename} ({size_kb:.1f} KB, SHA-256: {checksum[:12]}...)")

        # Rotate old backups
        rotate_backups(target_dir, retention_count)
        return backup_path
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        return None


def rotate_backups(backup_dir: Path, retention_count: int) -> None:
    """Prune oldest backups exceeding the retention count."""
    backups = sorted(
        backup_dir.glob("zenova_db_backup_*.db.gz"),
        key=lambda p: p.stat().st_mtime
    )
    if len(backups) > retention_count:
        excess = len(backups) - retention_count
        logger.info(f"Pruning {excess} old backup(s) to enforce retention limit of {retention_count}...")
        for old_backup in backups[:excess]:
            try:
                old_backup.unlink()
                chk = backup_dir / f"{old_backup.name}.sha256"
                if chk.exists():
                    chk.unlink()
                logger.info(f"Deleted old backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Failed to delete old backup {old_backup}: {e}")


if __name__ == "__main__":
    retention = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    res = backup_database(retention_count=retention)
    sys.exit(0 if res else 1)
