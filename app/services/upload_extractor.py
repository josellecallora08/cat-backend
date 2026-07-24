"""Content extraction service for uploaded training documents.

Provides format-specific text extractors for PDF, DOCX, TXT, CSV, and MD files,
plus a SHA-256 content hash utility.
"""

from pathlib import Path
import csv
import hashlib
import io
import logging

import chardet
import docx
import pdfplumber

logger = logging.getLogger(__name__)


def extract_pdf(file_path: Path) -> str:
    """Extract text content from a PDF file using pdfplumber.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages, joined with newlines.
        Returns empty string if extraction fails.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n".join(pages_text)
    except Exception as e:
        logger.error("Failed to extract PDF content from %s: %s", file_path, e)
        return ""


def extract_docx(file_path: Path) -> str:
    """Extract text content from a DOCX file using python-docx.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Text from all paragraphs, joined with newlines.
    """
    document = docx.Document(str(file_path))
    paragraphs = [para.text for para in document.paragraphs]
    return "\n".join(paragraphs)


def extract_txt(file_path: Path) -> str:
    """Extract text content from a plain text file with encoding detection.

    Uses chardet to detect the file encoding, falling back to UTF-8
    with error replacement if detection fails.

    Args:
        file_path: Path to the text file.

    Returns:
        Decoded text content.
    """
    raw_bytes = file_path.read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw_bytes.decode("utf-8", errors="replace")


def extract_csv(file_path: Path) -> str:
    """Extract text content from a CSV file.

    Uses chardet for encoding detection, then parses with stdlib csv.reader.
    Rows are separated by newlines, cells by commas.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Text representation with rows separated by newlines and cells by commas.
    """
    raw_bytes = file_path.read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"
    try:
        text = raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        text = raw_bytes.decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(text))
    rows = [",".join(row) for row in reader]
    return "\n".join(rows)


def extract_md(file_path: Path) -> str:
    """Extract raw markdown text from a file.

    Reads file bytes and decodes as UTF-8 with error replacement.
    No markdown processing is applied.

    Args:
        file_path: Path to the markdown file.

    Returns:
        Raw markdown text content.
    """
    raw_bytes = file_path.read_bytes()
    return raw_bytes.decode("utf-8", errors="replace")


def extract_content(file_path: Path, extension: str) -> str:
    """Dispatch to the appropriate format-specific extractor.

    Args:
        file_path: Path to the uploaded file.
        extension: File extension including the dot (e.g., '.pdf').

    Returns:
        Extracted text content.

    Raises:
        ValueError: If the extension is not supported.
    """
    extractors = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".txt": extract_txt,
        ".csv": extract_csv,
        ".md": extract_md,
    }

    ext_lower = extension.lower()
    extractor = extractors.get(ext_lower)
    if extractor is None:
        raise ValueError(f"Unsupported file extension: {extension}")

    return extractor(file_path)


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of the content.

    Args:
        content: Text content to hash.

    Returns:
        Hex digest string of the SHA-256 hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
