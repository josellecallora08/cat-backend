"""Upload validation pipeline for secure document uploads.

Validates file extension, MIME type, binary signature, file size (streaming),
DOCX archive safety (zip-bomb, path traversal, macros, encryption, symlinks),
and PDF encryption detection (using pypdf).

Does NOT perform malware scanning or content extraction.
"""

import os
import stat
import zipfile
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

from app.config import settings


class UploadRejectionReason(str, Enum):
    """Enumeration of all possible upload rejection reasons."""

    INVALID_EXTENSION = "invalid_extension"
    MIME_MISMATCH = "mime_extension_mismatch"
    SIGNATURE_MISMATCH = "signature_mismatch"
    FILE_TOO_LARGE = "file_too_large"
    FILE_EMPTY = "file_empty"
    DOCX_TOO_MANY_ENTRIES = "docx_too_many_entries"
    DOCX_TOO_LARGE_UNCOMPRESSED = "docx_too_large_uncompressed"
    DOCX_TOO_DEEP = "docx_too_deep"
    DOCX_MALFORMED = "docx_malformed"
    DOCX_MACRO_ENABLED = "docx_macro_enabled"
    DOCX_ENCRYPTED = "docx_encrypted"
    DOCX_UNSAFE_CONTAINER = "docx_unsafe_container"
    DOCX_ZIP_SLIP = "docx_zip_slip"
    DOCX_SUSPICIOUS_ENTRY = "docx_suspicious_entry"
    DOCX_EXCESSIVE_RATIO = "docx_excessive_compression_ratio"
    DOCX_NESTED_ARCHIVE = "docx_nested_archive"
    PDF_ENCRYPTED = "pdf_encrypted"
    PDF_MALFORMED = "pdf_malformed"
    MALWARE_DETECTED = "malware_detected"
    SCANNER_UNAVAILABLE = "scanner_unavailable"
    RATE_LIMITED = "rate_limited"


# OLE/CFB compound file signature (encrypted Office documents)
_OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"

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
        "magic_bytes": None,
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

_BINARY_SIGNATURES = [
    b"\x25\x50\x44\x46",  # PDF
    b"\x50\x4B\x03\x04",  # ZIP/DOCX
    b"\x4D\x5A",  # EXE
    b"\x7f\x45\x4c\x46",  # ELF
    _OLE_SIGNATURE,  # OLE
]

_SUSPICIOUS_EXTENSIONS = {
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js",
    ".com", ".scr", ".pif", ".msi", ".hta",
}

_MAX_COMPRESSION_RATIO = 100


def validate_extension(filename: str) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate file extension against the allowed whitelist."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_FORMATS:
        return (False, UploadRejectionReason.INVALID_EXTENSION)
    return (True, None)


def validate_mime_type(
    filename: str, content_type: str
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate MIME type matches allowed types for the extension."""
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
    """Validate binary signature matches the declared extension.

    For .docx files: also detects OLE/CFB containers (encrypted Office docs).
    For binary formats: checks magic bytes prefix.
    For text formats: verifies valid UTF-8 and no binary signatures.
    """
    if not header_bytes:
        return (False, UploadRejectionReason.FILE_EMPTY)

    ext = os.path.splitext(filename)[1].lower()
    format_info = ALLOWED_FORMATS.get(ext)
    if format_info is None:
        return (False, UploadRejectionReason.INVALID_EXTENSION)

    magic_bytes = format_info["magic_bytes"]

    if ext == ".docx":
        # Detect OLE/CFB containers: valid unencrypted DOCX must use ZIP/OOXML.
        # Any .docx with OLE signature is unsafe (encrypted or legacy format).
        if header_bytes.startswith(_OLE_SIGNATURE):
            return (False, UploadRejectionReason.DOCX_UNSAFE_CONTAINER)
        # Normal ZIP signature check
        if not header_bytes.startswith(magic_bytes):
            return (False, UploadRejectionReason.SIGNATURE_MISMATCH)
    elif magic_bytes is not None:
        if not header_bytes.startswith(magic_bytes):
            return (False, UploadRejectionReason.SIGNATURE_MISMATCH)
    else:
        # Text format: verify valid UTF-8 and no binary signatures
        try:
            header_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return (False, UploadRejectionReason.SIGNATURE_MISMATCH)
        for sig in _BINARY_SIGNATURES:
            if header_bytes.startswith(sig):
                return (False, UploadRejectionReason.SIGNATURE_MISMATCH)

    return (True, None)


async def validate_file_size_streaming(
    upload_file, max_size: int
) -> Tuple[bytes, Optional[UploadRejectionReason]]:
    """Read upload in 64KB streaming chunks; reject if exceeds max_size or empty."""
    chunk_size = 65536
    buffer = bytearray()

    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_size:
            return (bytes(buffer), UploadRejectionReason.FILE_TOO_LARGE)

    if len(buffer) == 0:
        return (bytes(buffer), UploadRejectionReason.FILE_EMPTY)

    return (bytes(buffer), None)


def validate_pdf_not_encrypted(file_bytes: bytes) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Check if a PDF is encrypted using pypdf.

    Uses pypdf.PdfReader.is_encrypted for reliable detection.
    Falls back to fail-closed on malformed PDFs.
    """
    import io
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            return (False, UploadRejectionReason.PDF_ENCRYPTED)
    except Exception:
        # Malformed PDF — fail closed
        return (False, UploadRejectionReason.PDF_MALFORMED)
    return (True, None)


def _is_unix_symlink(info: zipfile.ZipInfo) -> bool:
    """Correctly detect Unix symlink entries in ZIP archives.

    Only checks entries created on Unix systems (create_system == 3).
    Extracts the Unix file-type bits from external_attr and compares
    against stat.S_IFLNK (0o120000).
    """
    # create_system: 0 = Windows, 3 = Unix
    if info.create_system != 3:
        return False
    # Unix mode is stored in the upper 16 bits of external_attr
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    # Check if the file-type bits indicate a symlink
    return stat.S_ISLNK(unix_mode)


def validate_docx_archive(
    file_path: Path,
) -> Tuple[bool, Optional[UploadRejectionReason]]:
    """Validate a DOCX file's ZIP archive structure for safety.

    Checks:
    - Valid ZIP container (not malformed)
    - Not macro-enabled (DOCM/VBA content)
    - Not encrypted (EncryptionInfo/EncryptedPackage entries)
    - Entry count within limits
    - Total uncompressed size within limits
    - Compression ratio not excessive (zip bomb detection)
    - Directory nesting depth within limits
    - No path traversal (../ or absolute paths)
    - No symlinks (correct Unix mode detection)
    - No suspicious embedded executables
    - No nested archive files

    Depth algorithm: counts directory components using PurePosixPath.parts.
    'word/_rels/document.xml.rels' → parts=('word','_rels','document.xml.rels')
    → dir_depth = 2 (number of directory parts, excluding the filename).
    Standard DOCX files have entries at dir_depth 0-2, which are all accepted
    with the default max_depth=2 (rejects only when dir_depth > max_depth).
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            info_list = zf.infolist()

            # Check for macro-enabled content
            for info in info_list:
                name_lower = info.filename.lower()
                if name_lower == "[content_types].xml":
                    try:
                        ct = zf.read(info.filename).decode("utf-8", errors="replace").lower()
                        if "vnd.ms-office.vbaproject" in ct:
                            return (False, UploadRejectionReason.DOCX_MACRO_ENABLED)
                        if "macroenabledtemplate" in ct:
                            return (False, UploadRejectionReason.DOCX_MACRO_ENABLED)
                    except Exception:
                        pass
                if "vbaproject" in name_lower:
                    return (False, UploadRejectionReason.DOCX_MACRO_ENABLED)

            # Check for encryption markers
            for info in info_list:
                nl = info.filename.lower()
                if nl in ("encryptedpackage", "encryptioninfo"):
                    return (False, UploadRejectionReason.DOCX_ENCRYPTED)

            # Entry count
            if len(info_list) > settings.upload_max_docx_entries:
                return (False, UploadRejectionReason.DOCX_TOO_MANY_ENTRIES)

            # Total uncompressed size
            total_uncompressed = sum(i.file_size for i in info_list)
            if total_uncompressed > settings.upload_max_docx_uncompressed_size:
                return (False, UploadRejectionReason.DOCX_TOO_LARGE_UNCOMPRESSED)

            # Compression ratio
            total_compressed = sum(i.compress_size for i in info_list)
            if total_compressed > 0:
                ratio = total_uncompressed / total_compressed
                if ratio > _MAX_COMPRESSION_RATIO:
                    return (False, UploadRejectionReason.DOCX_EXCESSIVE_RATIO)

            # Per-entry checks
            max_depth = settings.upload_max_docx_depth
            for info in info_list:
                entry_path = info.filename

                # ZIP-slip: path traversal
                if ".." in entry_path or entry_path.startswith("/"):
                    return (False, UploadRejectionReason.DOCX_ZIP_SLIP)

                # Windows absolute path
                if len(entry_path) >= 2 and entry_path[1] == ":":
                    return (False, UploadRejectionReason.DOCX_ZIP_SLIP)

                # Correct symlink detection (Unix mode bits)
                if _is_unix_symlink(info):
                    return (False, UploadRejectionReason.DOCX_SUSPICIOUS_ENTRY)

                # Directory depth
                parts = PurePosixPath(entry_path).parts
                if entry_path.endswith("/"):
                    dir_depth = len(parts)
                else:
                    dir_depth = len(parts) - 1

                if dir_depth > max_depth:
                    return (False, UploadRejectionReason.DOCX_TOO_DEEP)

                # Suspicious extensions
                entry_ext = os.path.splitext(entry_path)[1].lower()
                if entry_ext in _SUSPICIOUS_EXTENSIONS:
                    return (False, UploadRejectionReason.DOCX_SUSPICIOUS_ENTRY)

                # Nested archives
                if entry_ext in {".zip", ".rar", ".7z", ".tar", ".gz"}:
                    return (False, UploadRejectionReason.DOCX_NESTED_ARCHIVE)

    except zipfile.BadZipFile:
        return (False, UploadRejectionReason.DOCX_MALFORMED)
    except Exception:
        return (False, UploadRejectionReason.DOCX_MALFORMED)

    return (True, None)
