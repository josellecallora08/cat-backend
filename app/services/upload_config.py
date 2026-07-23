"""Upload validation module.

Provides file type, size, and archive validation utilities
for secure file upload handling.
"""

import os
import re
import threading
import zipfile
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.config import settings


class UploadRejectionReason(str, Enum):
    """Reasons a file upload may be rejected."""

    INVALID_EXTENSION = "invalid_extension"
    INVALID_MIME_TYPE = "invalid_mime_type"
    MIME_EXTENSION_MISMATCH = "mime_extension_mismatch"
    FILE_TOO_LARGE = "file_too_large"
    ARCHIVE_TOO_MANY_FILES = "archive_too_many_files"
    ARCHIVE_TOO_LARGE = "archive_too_large_extracted"
    ARCHIVE_TOO_DEEP = "archive_too_deep"
    RATE_LIMITED = "rate_limited"


# Extension → set of valid MIME types
ALLOWED_MIME_TYPES: Dict[str, Set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".csv": {"text/csv"},
    ".txt": {"text/plain"},
    ".mp3": {"audio/mpeg"},
    ".wav": {"audio/wav", "audio/x-wav"},
    ".zip": {"application/zip", "application/x-zip-compressed"},
}


def validate_file_type(
    filename: str, content_type: str
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate that the file extension and MIME type are allowed.

    Args:
        filename: The original filename of the uploaded file.
        content_type: The MIME content type reported by the client.

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Get accepted extensions from settings
    accepted_extensions = [
        e.strip() for e in settings.upload_accepted_extensions.split(",")
    ]

    # Check if extension is in accepted list
    if ext not in accepted_extensions:
        return (False, UploadRejectionReason.INVALID_EXTENSION)

    # Check if extension is in ALLOWED_MIME_TYPES
    if ext not in ALLOWED_MIME_TYPES:
        return (False, UploadRejectionReason.INVALID_EXTENSION)

    # Check if content_type matches expected MIME types for this extension
    if content_type not in ALLOWED_MIME_TYPES[ext]:
        return (False, UploadRejectionReason.MIME_EXTENSION_MISMATCH)

    return (True, None)


def validate_file_size(size_bytes: int) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate that the file size does not exceed the configured maximum.

    Args:
        size_bytes: The size of the file in bytes.

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    if size_bytes > settings.upload_max_file_size:
        return (False, UploadRejectionReason.FILE_TOO_LARGE)

    return (True, None)


def validate_archive(
    file_path: Path,
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate a ZIP archive without extracting it.

    Checks the number of files, total extracted size, and nesting depth
    against configured limits.

    Args:
        file_path: Path to the ZIP archive file.

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    with zipfile.ZipFile(file_path, "r") as zf:
        info_list = zf.infolist()

        # Check number of files
        if len(info_list) > settings.upload_max_archive_files:
            return (False, UploadRejectionReason.ARCHIVE_TOO_MANY_FILES)

        # Check total uncompressed size
        total_size = sum(entry.file_size for entry in info_list)
        if total_size > settings.upload_max_archive_extracted_size:
            return (False, UploadRejectionReason.ARCHIVE_TOO_LARGE)

        # Check nesting depth: look for nested .zip files beyond allowed depth
        max_depth = settings.upload_max_archive_depth
        for entry in info_list:
            # Count directory depth by counting path separators
            depth = entry.filename.count("/")
            # Check if entry is a nested zip beyond allowed depth
            if entry.filename.lower().endswith(".zip") and depth >= max_depth:
                return (False, UploadRejectionReason.ARCHIVE_TOO_DEEP)

    return (True, None)


def get_quarantine_path() -> Path:
    """Get the quarantine directory path, creating it if necessary.

    Resolves the configured quarantine path relative to the project root
    and creates the directory with restricted permissions (0o700) if it
    does not already exist.

    Returns:
        The resolved Path to the quarantine directory.
    """
    quarantine_path = Path(settings.upload_quarantine_path).resolve()
    quarantine_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return quarantine_path


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection attacks.

    Strips path separators, removes null bytes and control characters,
    collapses multiple dots, and limits length to 255 characters.

    Args:
        filename: The original filename to sanitize.

    Returns:
        The sanitized filename safe for filesystem use.
    """
    # Strip path separators and parent directory references
    filename = filename.replace("/", "").replace("\\", "").replace("..", "")

    # Remove null bytes and control characters (< 0x20)
    filename = "".join(ch for ch in filename if ord(ch) >= 0x20)
    filename = filename.replace("\x00", "")

    # Replace multiple consecutive dots with a single dot
    filename = re.sub(r"\.{2,}", ".", filename)

    # Limit length to 255 characters
    filename = filename[:255]

    return filename


# --- Rejection Rate Tracking ---

# In-memory sliding window tracker: user_id → list of rejection timestamps
_rejection_tracker: Dict[str, List[datetime]] = {}
_tracker_lock = threading.Lock()


def record_rejection(user_id: str) -> None:
    """Record a file rejection event for the given user.

    Args:
        user_id: The unique identifier of the user who triggered the rejection.
    """
    now = datetime.utcnow()
    with _tracker_lock:
        if user_id not in _rejection_tracker:
            _rejection_tracker[user_id] = []
        _rejection_tracker[user_id].append(now)


def is_rate_limited(user_id: str) -> bool:
    """Check if a user has exceeded the rejection rate limit.

    Uses a sliding window to count rejections within the configured
    time window and compares against the maximum allowed attempts.

    Args:
        user_id: The unique identifier of the user to check.

    Returns:
        True if the user is rate-limited, False otherwise.
    """
    now = datetime.utcnow()
    window = timedelta(minutes=settings.upload_rejection_window_minutes)
    cutoff = now - window

    with _tracker_lock:
        if user_id not in _rejection_tracker:
            return False

        # Filter to only recent entries within the window
        recent = [ts for ts in _rejection_tracker[user_id] if ts > cutoff]
        _rejection_tracker[user_id] = recent

        return len(recent) >= settings.upload_rejection_max_attempts


def cleanup_expired_entries() -> int:
    """Remove expired entries from the rejection tracker to prevent memory leaks.

    Removes all entries older than the configured rejection window for all users.
    Users with no remaining entries are removed entirely.

    Returns:
        The number of users cleaned up (fully removed from tracker).
    """
    now = datetime.utcnow()
    window = timedelta(minutes=settings.upload_rejection_window_minutes)
    cutoff = now - window
    cleaned = 0

    with _tracker_lock:
        users_to_remove = []
        for user_id, timestamps in _rejection_tracker.items():
            # Keep only timestamps within the window
            _rejection_tracker[user_id] = [ts for ts in timestamps if ts > cutoff]
            if not _rejection_tracker[user_id]:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del _rejection_tracker[user_id]
            cleaned += 1

    return cleaned
