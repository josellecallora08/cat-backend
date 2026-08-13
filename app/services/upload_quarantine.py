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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import settings
from app.models.script_upload import ScriptUpload


logger = logging.getLogger(__name__)

# Strict extension whitelist for quarantine storage
_ALLOWED_QUARANTINE_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".csv", ".md"})


@dataclass
class CleanupResult:
    """The outcomes from one sequential quarantine cleanup pass."""

    deleted: int = 0
    orphaned: int = 0
    skipped: int = 0
    errors: int = 0


def validate_retention_hours(value: int) -> int:
    """Clamp a dynamically supplied retention value to the supported range."""
    if value < 0:
        logger.warning("Negative retention_hours (%d) treated as 0", value)
        return 0
    if value > 8760:
        logger.warning("Retention_hours (%d) clamped to 8760", value)
        return 8760
    return value


def validate_cleanup_interval_minutes(value: int) -> int:
    """Clamp a dynamically supplied scheduler interval to the supported range."""
    if value < 1:
        logger.warning("Cleanup interval (%d) clamped to 1 minute", value)
        return 1
    if value > 1440:
        logger.warning("Cleanup interval (%d) clamped to 1440 minutes", value)
        return 1440
    return value


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


async def cleanup_expired_files(session_factory: async_sessionmaker) -> CleanupResult:
    """Purge expired raw files while retaining their database audit history.

    The database transaction is committed before the file is removed.  A database
    failure therefore leaves the raw file intact for a later cleanup pass.
    """
    quarantine_dir = get_quarantine_path()
    retention_hours = validate_retention_hours(
        settings.upload_quarantine_retention_hours
    )
    retention_seconds = retention_hours * 3600
    now = time.time()
    result = CleanupResult()
    logger.debug(
        "Starting quarantine cleanup pass: retention_hours=%d utc=%s",
        retention_hours,
        datetime.now(UTC).isoformat(),
        extra={
            "event": "quarantine_cleanup_started",
            "retention_hours": retention_hours,
            "cleanup_started_at": datetime.now(UTC).isoformat(),
        },
    )

    for file_path in quarantine_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.name == ".gitkeep" or file_path.name.startswith(
            ".tmp_upload_"
        ):
            continue

        try:
            mtime = file_path.stat().st_mtime
        except FileNotFoundError:
            logger.warning(
                "Quarantine file already removed: %s",
                file_path.name,
                extra={
                    "event": "quarantine_file_already_removed",
                    "storage_key": file_path.name,
                },
            )
            result.skipped += 1
            continue
        except OSError as error:
            logger.warning(
                "Unable to inspect quarantine file %s: %s",
                file_path.name,
                error,
                extra={
                    "event": "quarantine_file_inspection_failed",
                    "storage_key": file_path.name,
                    "failure_reason": str(error),
                },
            )
            result.skipped += 1
            result.errors += 1
            continue
        age_seconds = now - mtime

        # A zero-hour retention explicitly makes every completed quarantine
        # file eligible, including files with clock-skewed future mtimes.
        if retention_hours != 0 and age_seconds <= retention_seconds:
            continue

        try:
            async with session_factory() as session:
                upload = (
                    await session.execute(
                        select(ScriptUpload).where(
                            ScriptUpload.storage_key == file_path.name
                        )
                    )
                ).scalar_one_or_none()

                if upload is not None and upload.deleted_at is None:
                    deletion_time = datetime.now(UTC)
                    await session.execute(
                        update(ScriptUpload)
                        .where(ScriptUpload.id == upload.id)
                        .values(
                            deleted_at=deletion_time,
                            # ScriptUpload.updated_at has an automatic on-update
                            # value. Explicitly retaining the column value prevents
                            # cleanup from modifying audit metadata.
                            updated_at=ScriptUpload.updated_at,
                        )
                    )
                    await session.commit()
        except Exception as error:
            logger.error(
                "Failed to record quarantine deletion for %s: %s",
                file_path.name,
                error,
                exc_info=True,
                extra={
                    "event": "quarantine_metadata_update_failed",
                    "storage_key": file_path.name,
                    "failure_reason": str(error),
                },
            )
            result.skipped += 1
            result.errors += 1
            continue

        try:
            file_path.unlink()
        except FileNotFoundError:
            logger.warning(
                "Quarantine file already removed: %s",
                file_path.name,
                extra={
                    "event": "quarantine_file_already_removed",
                    "storage_key": file_path.name,
                },
            )
            result.skipped += 1
        except OSError as error:
            logger.warning(
                "Failed to delete quarantine file %s: %s",
                file_path.name,
                error,
                extra={
                    "event": "quarantine_file_deletion_failed",
                    "storage_key": file_path.name,
                    "failure_reason": str(error),
                },
            )
            result.skipped += 1
            result.errors += 1
        else:
            if upload is None:
                result.orphaned += 1
                logger.warning(
                    "Removed orphaned quarantine file: %s",
                    file_path.name,
                    extra={
                        "event": "orphaned_quarantine_file_removed",
                        "storage_key": file_path.name,
                    },
                )
            else:
                result.deleted += 1
            logger.debug(
                "Deleted expired quarantine file: %s",
                file_path.name,
                extra={
                    "event": "quarantine_file_deleted",
                    "storage_key": file_path.name,
                },
            )

    logger.info(
        "Quarantine cleanup complete: deleted=%d orphaned=%d skipped=%d total_removed=%d",
        result.deleted,
        result.orphaned,
        result.skipped,
        result.deleted + result.orphaned,
        extra={
            "event": "quarantine_cleanup_completed",
            "deleted": result.deleted,
            "orphaned": result.orphaned,
            "skipped": result.skipped,
            "errors": result.errors,
            "total_removed": result.deleted + result.orphaned,
        },
    )
    return result


async def start_cleanup_scheduler(app) -> None:
    """Run initial cleanup, then start the recurring background task."""
    from app.database import async_session_factory

    async def _cleanup_loop() -> None:
        try:
            while True:
                # Read settings for every cycle so runtime configuration changes
                # take effect without restarting the application.
                interval_seconds = validate_cleanup_interval_minutes(
                    settings.upload_quarantine_cleanup_interval_minutes
                ) * 60
                await asyncio.sleep(interval_seconds)
                await cleanup_expired_files(async_session_factory)
        except asyncio.CancelledError:
            logger.info("Quarantine cleanup scheduler cancelled")
            raise

    # Startup does not complete until the required initial pass has completed.
    await cleanup_expired_files(async_session_factory)
    task = asyncio.create_task(_cleanup_loop())
    app.state.quarantine_cleanup_task = task


async def stop_cleanup_scheduler(app, timeout_seconds: float = 5) -> bool:
    """Cancel the recurring cleanup task without blocking shutdown indefinitely.

    Returns:
        True when the task stopped within the timeout, otherwise False.
    """
    task = getattr(app.state, "quarantine_cleanup_task", None)
    if task is None or task.done():
        return True

    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except asyncio.CancelledError:
        return True
    except TimeoutError:
        logger.warning(
            "Timed out waiting for quarantine cleanup scheduler cancellation"
        )
        return False
    return True
