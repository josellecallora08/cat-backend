"""Quarantine storage, file sanitization, and automatic cleanup of expired files.

This module manages the quarantine lifecycle for uploaded files:
- Stores uploaded files with UUID-based filenames in a restricted directory.
- Sanitizes user-provided filenames to prevent path traversal and injection.
- Periodically cleans up files that exceed the configured retention period.
"""

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def get_quarantine_path() -> Path:
    """Resolve and create the quarantine directory with restricted permissions.

    Returns:
        Path: Absolute path to the quarantine directory (mode 0o700).
    """
    quarantine_dir = Path(settings.upload_quarantine_path).resolve()
    quarantine_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return quarantine_dir


def store_in_quarantine(file_bytes: bytes, extension: str) -> Path:
    """Store file bytes in quarantine with a UUID4-based filename.

    Args:
        file_bytes: Raw file content to store.
        extension: File extension including the dot (e.g., '.pdf').

    Returns:
        Path: Full path to the stored quarantine file.
    """
    uuid_filename = f"{uuid.uuid4()}{extension}"
    quarantine_dir = get_quarantine_path()
    file_path = quarantine_dir / uuid_filename
    file_path.write_bytes(file_bytes)
    return file_path


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection attacks.

    Strips path separators, removes '..' sequences, null bytes, control
    characters, and collapses multiple consecutive dots to a single dot.
    Limits the result to 255 characters.

    Args:
        filename: The raw user-provided filename.

    Returns:
        str: Sanitized filename safe for filesystem use.
    """
    # Strip path separators
    filename = filename.replace("/", "").replace("\\", "")

    # Remove '..' sequences
    filename = filename.replace("..", "")

    # Remove null bytes and control characters (ord < 0x20)
    filename = "".join(c for c in filename if c != "\x00" and ord(c) >= 0x20)

    # Collapse multiple consecutive dots to a single dot
    filename = re.sub(r"\.{2,}", ".", filename)

    # Limit length to 255 characters
    filename = filename[:255]

    return filename


def cleanup_expired_files() -> int:
    """Delete quarantine files older than the configured retention period.

    Iterates all files in the quarantine directory and removes any whose
    modification time exceeds `upload_quarantine_retention_hours`.

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

        mtime = file_path.stat().st_mtime
        age_seconds = now - mtime

        if age_seconds > retention_seconds:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug(
                    "Deleted expired quarantine file: %s (age: %.1f hours)",
                    file_path.name,
                    age_seconds / 3600,
                )
            except OSError as e:
                logger.warning(
                    "Failed to delete quarantine file %s: %s",
                    file_path.name,
                    e,
                )

    logger.info(
        "Quarantine cleanup complete: %d file(s) deleted", deleted_count
    )
    return deleted_count


async def start_cleanup_scheduler(app) -> None:
    """Start a background asyncio task that periodically cleans expired quarantine files.

    Runs cleanup once immediately on startup, then sleeps for the configured
    interval between subsequent runs. The task reference is stored on
    `app.state.quarantine_cleanup_task` for graceful shutdown.

    Args:
        app: The FastAPI application instance.
    """
    interval_seconds = settings.upload_quarantine_cleanup_interval_minutes * 60

    async def _cleanup_loop() -> None:
        try:
            # Run cleanup once immediately on startup
            cleanup_expired_files()

            while True:
                await asyncio.sleep(interval_seconds)
                cleanup_expired_files()
        except asyncio.CancelledError:
            logger.info("Quarantine cleanup scheduler cancelled")
            raise

    task = asyncio.create_task(_cleanup_loop())
    app.state.quarantine_cleanup_task = task
