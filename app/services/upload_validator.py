"""Upload validation pipeline for secure document uploads.

This module provides the validation logic for the document upload pipeline,
replacing the old upload_config.py with a corrected whitelist and proper
binary signature validation.
"""

import os
import zipfile
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from app.config import settings


class UploadRejectionReason(str, Enum):
    """Enumeration of all possible upload rejection reasons."""

    INVALID_EXTENSION = "invalid_extension"
    MIME_MISMATCH = "mime_extension_mismatch"
    SIGNATURE_MISMATCH = "signature_mismatch"
    FILE_TOO_LARGE = "file_too_large"
    DOCX_TOO_MANY_ENTRIES = "docx_too_many_entries"
    DOCX_TOO_LARGE_UNCOMPRESSED = "docx_too_large_uncompressed"
    DOCX_TOO_DEEP = "docx_too_deep"
    MALWARE_DETECTED = "malware_detected"
    SCANNER_UNAVAILABLE = "scanner_unavailable"
    RATE_LIMITED = "rate_limited"


ALLOWED_FORMATS = {
    ".pdf": {
        "mime_types": {"application/pdf"},
        "magic_bytes": b"\x25\x50\x44\x46",  # %PDF
    },
    ".docx": {
        "mime_types": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        },
        "magic_bytes": b"\x50\x4B\x03\x04",  # PK (ZIP header)
    },
    ".txt": {
        "mime_types": {"text/plain"},
        "magic_bytes": None,  # UTF-8/ASCII heuristic
    },
    ".csv": {
        "mime_types": {"text/csv"},
        "magic_bytes": None,
    },
    ".md": {
        "mime_types": {"text/markdown", "text/x-markdown"},
        "magic_bytes": None,
    },
}

# Known binary signatures used to detect misclassified binary files
_BINARY_SIGNATURES = [
    b"\x25\x50\x44\x46",  # PDF (%PDF)
    b"\x50\x4B\x03\x04",  # ZIP/DOCX (PK)
    b"\x4D\x5A",  # EXE (MZ)
]


def validate_extension(filename: str) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate that the file extension is in the allowed whitelist.

    Args:
        filename: The original filename including extension.

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_FORMATS:
        return (False, UploadRejectionReason.INVALID_EXTENSION)
    return (True, None)


def validate_mime_type(
    filename: str, content_type: str
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate that the content type matches the allowed MIME types for the file extension.

    Args:
        filename: The original filename including extension.
        content_type: The MIME content type declared by the client.

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    ext = os.path.splitext(filename)[1].lower()
    format_info = ALLOWED_FORMATS.get(ext)
    if format_info is None:
        return (False, UploadRejectionReason.INVALID_EXTENSION)
    if content_type not in format_info["mime_types"]:
        return (False, UploadRejectionReason.MIME_MISMATCH)
    return (True, None)


def validate_file_signature(
    filename: str, header_bytes: bytes
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate the file's binary signature (magic bytes) matches the declared extension.

    For formats with known magic bytes (PDF, DOCX), checks that the file header
    starts with the expected prefix. For text formats (.txt, .csv, .md), verifies
    that the content is valid UTF-8 and does NOT start with known binary signatures.

    Args:
        filename: The original filename including extension.
        header_bytes: The first bytes of the file content (at least 8 bytes recommended).

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    ext = os.path.splitext(filename)[1].lower()
    format_info = ALLOWED_FORMATS.get(ext)
    if format_info is None:
        return (False, UploadRejectionReason.INVALID_EXTENSION)

    magic_bytes = format_info["magic_bytes"]

    if magic_bytes is not None:
        # Binary format: check magic bytes prefix
        if not header_bytes.startswith(magic_bytes):
            return (False, UploadRejectionReason.SIGNATURE_MISMATCH)
    else:
        # Text format: verify valid UTF-8 and no binary signatures
        try:
            header_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return (False, UploadRejectionReason.SIGNATURE_MISMATCH)

        # Check against known binary signatures
        for sig in _BINARY_SIGNATURES:
            if header_bytes.startswith(sig):
                return (False, UploadRejectionReason.SIGNATURE_MISMATCH)

    return (True, None)


async def validate_file_size_streaming(
    upload_file, max_size: int
) -> Tuple[bytes, Optional[UploadRejectionReason]]:
    """Read the upload file in streaming chunks and enforce a size limit.

    Reads from the upload file in 64KB chunks, accumulating into a buffer.
    If the cumulative size exceeds max_size, reading stops and the file is
    rejected as too large.

    Args:
        upload_file: A file-like object with an async read(size) method
                     (e.g., FastAPI's UploadFile).
        max_size: Maximum allowed file size in bytes.

    Returns:
        A tuple of (file_bytes, rejection_reason). If the file is within the
        size limit, rejection_reason is None and file_bytes contains the full
        content. If the file exceeds the limit, file_bytes contains what was
        read so far and rejection_reason is FILE_TOO_LARGE.
    """
    chunk_size = 65536  # 64 KB
    buffer = bytearray()

    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_size:
            return (bytes(buffer), UploadRejectionReason.FILE_TOO_LARGE)

    return (bytes(buffer), None)


def validate_docx_archive(
    file_path: Path,
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate a DOCX file's ZIP archive structure for safety.

    Checks the ZIP archive metadata to guard against zip bombs and
    excessively nested structures:
    - Entry count must not exceed upload_max_docx_entries.
    - Total uncompressed size must not exceed upload_max_docx_uncompressed_size.
    - Nesting depth (path separators in entry names) must not exceed upload_max_docx_depth.

    Args:
        file_path: Path to the DOCX file on disk.

    Returns:
        A tuple of (is_valid, rejection_reason). If valid, rejection_reason is None.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            info_list = zf.infolist()

            # Check entry count
            if len(info_list) > settings.upload_max_docx_entries:
                return (False, UploadRejectionReason.DOCX_TOO_MANY_ENTRIES)

            # Check total uncompressed size
            total_uncompressed = sum(info.file_size for info in info_list)
            if total_uncompressed > settings.upload_max_docx_uncompressed_size:
                return (False, UploadRejectionReason.DOCX_TOO_LARGE_UNCOMPRESSED)

            # Check nesting depth
            max_depth = settings.upload_max_docx_depth
            for info in info_list:
                depth = info.filename.count("/")
                if depth >= max_depth:
                    return (False, UploadRejectionReason.DOCX_TOO_DEEP)

    except zipfile.BadZipFile:
        return (False, UploadRejectionReason.SIGNATURE_MISMATCH)

    return (True, None)
