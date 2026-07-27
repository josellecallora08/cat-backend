"""Content extraction service for uploaded training documents.

Extracts text safely from PDF (page text only), DOCX (body paragraphs only),
TXT, CSV, and MD files. Fails closed on malformed/unreadable documents.
"""

import csv
import hashlib
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Resource limits
_MAX_PAGES = 500
_MAX_EXTRACTED_BYTES = 5 * 1024 * 1024  # 5 MB of extracted text


class ExtractionError(Exception):
    """Raised when content extraction fails due to malformed or unsafe content."""
    pass


def extract_pdf(file_path: Path) -> str:
    """Extract page text only from a PDF. Fails closed on errors.

    Does not extract attachments, annotations, embedded objects, or metadata.
    Raises ExtractionError on failure (does NOT return empty string silently).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ExtractionError("pdfplumber not available")

    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) > _MAX_PAGES:
                raise ExtractionError(
                    f"PDF has {len(pdf.pages)} pages, exceeds limit of {_MAX_PAGES}"
                )
            pages_text = []
            total_size = 0
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    total_size += len(text.encode("utf-8"))
                    if total_size > _MAX_EXTRACTED_BYTES:
                        raise ExtractionError("Extracted text exceeds size limit")
                    pages_text.append(text)
            return "\n".join(pages_text)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"PDF extraction failed: {e}")


def extract_docx(file_path: Path) -> str:
    """Extract body paragraph text only from a DOCX.

    Does not extract: comments, headers/footers with external content,
    embedded objects, OLE objects, macros, relationships, or metadata.
    Raises ExtractionError on failure.
    """
    try:
        import docx as python_docx
    except ImportError:
        raise ExtractionError("python-docx not available")

    try:
        document = python_docx.Document(str(file_path))
        paragraphs = []
        total_size = 0
        for para in document.paragraphs:
            text = para.text
            if text:
                total_size += len(text.encode("utf-8"))
                if total_size > _MAX_EXTRACTED_BYTES:
                    raise ExtractionError("Extracted text exceeds size limit")
                paragraphs.append(text)
        return "\n".join(paragraphs)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"DOCX extraction failed: {e}")


def extract_txt(file_path: Path) -> str:
    """Extract text with encoding detection. Fails closed on read errors."""
    try:
        import chardet
    except ImportError:
        raise ExtractionError("chardet not available")

    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > _MAX_EXTRACTED_BYTES:
            raise ExtractionError("File exceeds extraction size limit")
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return raw_bytes.decode("utf-8", errors="replace")
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"TXT extraction failed: {e}")


def extract_csv(file_path: Path) -> str:
    """Extract CSV as text. Uses chardet for encoding detection."""
    try:
        import chardet
    except ImportError:
        raise ExtractionError("chardet not available")

    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > _MAX_EXTRACTED_BYTES:
            raise ExtractionError("File exceeds extraction size limit")
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        try:
            text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("utf-8", errors="replace")

        reader = csv.reader(io.StringIO(text))
        rows = [",".join(row) for row in reader]
        return "\n".join(rows)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"CSV extraction failed: {e}")


def extract_md(file_path: Path) -> str:
    """Extract raw markdown text as UTF-8."""
    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > _MAX_EXTRACTED_BYTES:
            raise ExtractionError("File exceeds extraction size limit")
        return raw_bytes.decode("utf-8", errors="replace")
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"MD extraction failed: {e}")


def extract_content(file_path: Path, extension: str) -> str:
    """Dispatch to format-specific extractor. Raises ExtractionError on failure."""
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
    """Compute SHA-256 hash of the extracted content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
