"""Quarantine storage with atomic writes, extension whitelist, and cleanup.

Security invariants:
- Original filenames are never used in storage paths.
- Only whitelisted extensions are accepted.
- Files are written atomically (temp → rename) to prevent partial files.
- Resolved paths are verified to stay inside the quarantine directory using
  Path.is_relative_to() (not string prefix comparison).
- Restricted permissions are applied where the OS supports them.
"""

import asyncio
import logging
import os
import platform
import re
import tempfile
import time
import uuid
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Strict extension whitelist for quarantine storage
_ALLOWED_QUARANTINE_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".csv", ".md"})


def get_quarantine_path() -> Path:
    """Resolve and create the quarantine directory with restricted permissions.

    Returns:
        Path: Absolute path to the quarantine directory.
    """
    quarantine_dir = Path(settings.upload_quarantine_path).resolve()
    quarantine_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return quarantine_dir


def store_in_quarantine(file_bytes: bytes, extension: str) -> Path:
    """Store file bytes in quarantine atomically with a random UUID filename.

    Workflow:
    1. Validate extension against strict whitelist.
    2. Create a temporary file inside the quarantine directory.
    3. Write all bytes and flush.
    4. Apply restricted permissions (Unix: 0o600).
    5. Atomically rename to final UUID-based filename.
    6. On any failure, delete the temporary file.

    Args:
        file_bytes: Raw file content to store.
        extension: Validated file extension (must start with dot).

    Returns:
        Path: Full path to the stored quarantine file.

    Raises:
        ValueError: If extension is not in the whitelist or path escapes quarantine.
    """
    # Strict extension whitelist
    clean_ext = extension.lower().strip() if extension else ""
    if clean_ext not in _ALLOWED_QUARANTINE_EXTENSIONS:
        raise ValueError(
            f"Extension '{extension}' not in quarantine whitelist: "
            f"{sorted(_ALLOWED_QUARANTINE_EXTENSIONS)}"
        )

    quarantine_dir = get_quarantine_path()
    final_name = f"{uuid.uuid4()}{clean_ext}"
    final_path = (quarantine_dir / final_name).resolve()

    # Path containment check using is_relative_to (not string prefix)
    if not final_path.is_relative_to(quarantine_dir):
        raise ValueError("Storage path escapes quarantine directory")

    # Atomic write: temp file → complete write loop → fsync → permissions → rename
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(quarantine_dir), prefix=".tmp_upload_"
        )
        tmp_path = Path(tmp_path_str)

        # Write ALL bytes using a loop (os.write may return partial counts)
        total = len(file_bytes)
        offset = 0
        while offset < total:
            written = os.write(tmp_fd, file_bytes[offset:])
            if written <= 0:
                raise OSError(
                    f"os.write returned {written} at offset {offset}/{total}"
                )
            offset += written

        # Verify size before fsync
        os.fsync(tmp_fd)
        os.close(tmp_fd)
        tmp_fd = None

        # Confirm file size matches expected
        actual_size = tmp_path.stat().st_size
        if actual_size != total:
            raise OSError(
                f"Size mismatch: expected {total}, got {actual_size}"
            )

        # Apply restricted permissions on Unix
        if platform.system() != "Windows":
            os.chmod(tmp_path_str, 0o600)

        # Atomic rename
        tmp_path.rename(final_path)
        return final_path

    except Exception:
        # Cleanup: close fd if still open, remove temp file
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        # Also remove final_path if somehow partially created
        if final_path.exists():
            try:
                final_path.unlink()
            except OSError:
                pass
        raise


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection.

    Strips path separators, removes '..' sequences, null bytes, control
    characters, collapses multiple dots, and limits length.
    """
    filename = filename.replace("/", "").replace("\\", "")
    filename = filename.replace("..", "")
    filename = "".join(c for c in filename if c != "\x00" and ord(c) >= 0x20)
    filename = re.sub(r"\.{2,}", ".", filename)
    filename = filename[:255]
    return filename


def cleanup_expired_files() -> int:
    """Delete quarantine files older than the configured retention period.

    Returns:
        int: Number of files deleted.
    """
    quarantine_dir = get_quarantine_path()
    retention_seconds = settings.upload_quarantine_retention_hours * 3600
    now = time.time()
    deleted_count = 0

    for file_path in quarantine_dir.iterdir():
        if not file_path.is_file():
            continue
        # Skip .gitkeep
        if file_path.name == ".gitkeep":
            continue

        mtime = file_path.stat().st_mtime
        age_seconds = now - mtime

        if age_seconds > retention_seconds:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug("Deleted expired quarantine file: %s", file_path.name)
            except OSError as e:
                logger.warning("Failed to delete quarantine file %s: %s", file_path.name, e)

    logger.info("Quarantine cleanup complete: %d file(s) deleted", deleted_count)
    return deleted_count


async def start_cleanup_scheduler(app) -> None:
    """Start background cleanup task on app startup."""
    interval_seconds = settings.upload_quarantine_cleanup_interval_minutes * 60

    async def _cleanup_loop() -> None:
        try:
            cleanup_expired_files()
            while True:
                await asyncio.sleep(interval_seconds)
                cleanup_expired_files()
        except asyncio.CancelledError:
            logger.info("Quarantine cleanup scheduler cancelled")
            raise

    task = asyncio.create_task(_cleanup_loop())
    app.state.quarantine_cleanup_task = task
