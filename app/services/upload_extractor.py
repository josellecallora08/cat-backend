"""Content extraction service for uploaded training documents.

Extracts text safely from PDF, DOCX, TXT, CSV, and MD files.

Policies:
- BOM handling: Detected and consumed for UTF-8/16/32; decoded text never
  starts with U+FEFF from a BOM. Uses BOM-aware codecs (utf-8-sig, utf-16,
  utf-32) that consume the marker automatically.
- Encoding detection order: BOM > strict UTF-8 > chardet (confidence >= 0.5,
  rejecting UTF-7) > UTF-8 with replacement.
- Chardet confidence threshold: 0.5. Below this, falls back to UTF-8.
- CSV serialization: csv.writer with CRLF terminator. Size limit enforced
  per-row in UTF-8 bytes. Rejects (not truncates) when limit exceeded.
- CSV formula neutralization: Inspects first non-whitespace char (including
  after U+FEFF). Triggers: =, +, -, @, |. Also: literal tab as first char.
  Whitespace-only cells unchanged.
- Markdown URI policy: DENYLIST of dangerous schemes (data, javascript,
  vbscript, file). URI normalization: html.unescape + urllib percent-decode +
  control-char removal + case normalization, with 3-iteration bound.
  Unknown/custom schemes are allowed (denylist, not allowlist).
- Markdown reference-style links: Definitions are inspected with the same
  URI policy. Dangerous definitions have their destination replaced with #.
- Raw HTML policy: ALL raw HTML removed. Active elements (script, style,
  iframe, object, embed, svg, math) have content removed too.
"""

import codecs
import csv
import hashlib
import html
import io
import logging
import re
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

# Resource limits
_MAX_PAGES = 500
_MAX_EXTRACTED_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_CSV_ROWS = 100_000
_MAX_CSV_CELL_LEN = 32_768
_CHARDET_CONFIDENCE_THRESHOLD = 0.5

# CSV formula triggers (checked after stripping whitespace and U+FEFF)
_FORMULA_TRIGGERS = frozenset("=+-@|")

# Control chars to strip (C0 except \t\n\r, DEL, C1)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# ─── HTML sanitation ──────────────────────────────────────────────────────
_ACTIVE_ELEMENTS = {"script", "style", "iframe", "object", "embed", "svg", "math"}
_ACTIVE_ELEMENT_RE = re.compile(
    r"<\s*(" + "|".join(_ACTIVE_ELEMENTS) + r")\b[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SELF_CLOSING_ACTIVE_RE = re.compile(
    r"<\s*(" + "|".join(_ACTIVE_ELEMENTS) + r")\b[^>]*/?\s*>",
    re.IGNORECASE,
)
_UNCLOSED_ACTIVE_RE = re.compile(
    r"<\s*(" + "|".join(_ACTIVE_ELEMENTS) + r")\b[^>]*>.*",
    re.IGNORECASE | re.DOTALL,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]{0,2000}>", re.DOTALL)

# ─── Markdown links ──────────────────────────────────────────────────────
_MD_LINK_RE = re.compile(
    r"(!?\[[^\]]*\])"
    r"\("
    r"((?:[^()]*|\([^()]*\))*)"
    r"\)",
    re.DOTALL,
)
# Reference-style definition: [id]: URL "optional title"
_MD_REF_DEF_RE = re.compile(
    r"^(\s{0,3}\[[^\]]+\]:\s+)"  # [id]: (with up to 3 leading spaces)
    r"(\S+)"                      # destination
    r"(.*)?$",                    # optional title
    re.MULTILINE,
)

# Dangerous URI schemes (denylist)
_DANGEROUS_SCHEMES = {"data", "javascript", "vbscript", "file"}
_URI_NORMALIZATION_MAX_ITER = 3


class ExtractionError(Exception):
    """Raised when content extraction fails."""
    pass


# ─── Encoding ─────────────────────────────────────────────────────────────

def _normalize_encoding(raw_bytes: bytes) -> str:
    """Decode bytes to str. BOM > strict UTF-8 > chardet > replacement.

    BOMs are consumed (decoded text never starts with U+FEFF from a BOM).
    """
    if not raw_bytes:
        return ""

    # 1. BOM detection using BOM-aware codecs that consume the marker
    if raw_bytes.startswith(codecs.BOM_UTF32_LE):
        try:
            return raw_bytes.decode("utf-32")
        except (UnicodeDecodeError, LookupError):
            pass
    if raw_bytes.startswith(codecs.BOM_UTF32_BE):
        try:
            return raw_bytes.decode("utf-32")
        except (UnicodeDecodeError, LookupError):
            pass
    if raw_bytes.startswith(codecs.BOM_UTF16_LE) or raw_bytes.startswith(codecs.BOM_UTF16_BE):
        try:
            return raw_bytes.decode("utf-16")
        except (UnicodeDecodeError, LookupError):
            pass
    if raw_bytes.startswith(codecs.BOM_UTF8):
        try:
            return raw_bytes.decode("utf-8-sig")
        except (UnicodeDecodeError, LookupError):
            pass

    # 2. Strict UTF-8
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # 3. chardet with confidence threshold
    try:
        import chardet
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding")
        confidence = detected.get("confidence")
        if encoding and _is_chardet_trustworthy(encoding, confidence):
            try:
                return raw_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        pass

    # 4. Guaranteed fallback
    return raw_bytes.decode("utf-8", errors="replace")


def _is_chardet_trustworthy(encoding: str, confidence) -> bool:
    """Check if a chardet result should be trusted."""
    if not encoding:
        return False
    # Reject UTF-7 (frequent false positive)
    normalized = encoding.lower().replace("-", "").replace("_", "")
    if normalized == "utf7":
        return False
    # Confidence must be a number >= threshold
    if confidence is None:
        return False
    try:
        conf_val = float(confidence)
    except (TypeError, ValueError):
        return False
    return conf_val >= _CHARDET_CONFIDENCE_THRESHOLD


# ─── Text helpers ─────────────────────────────────────────────────────────

def _strip_control_chars(text: str) -> str:
    """Remove C0/C1 control characters, preserving tab, newline, CR."""
    return _CONTROL_CHAR_RE.sub("", text)


def _enforce_size_limit(text: str) -> str:
    """Truncate text if UTF-8 encoding exceeds limit."""
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_EXTRACTED_BYTES:
        return text
    return encoded[:_MAX_EXTRACTED_BYTES].decode("utf-8", errors="ignore")


# ─── CSV formula neutralization ───────────────────────────────────────────

def _neutralize_csv_cell(cell: str) -> str:
    """Neutralize formula triggers in a CSV cell.

    Detection: strip leading whitespace AND U+FEFF (ZWNBSP/BOM artifact) to
    find the first significant character. If it's a trigger, prefix the
    ORIGINAL cell with a single quote. Tab as first char also triggers.
    Whitespace-only cells are unchanged.
    Final length capped at _MAX_CSV_CELL_LEN.
    """
    if not cell:
        return cell

    # Tab as literal first char
    if cell[0] == "\t":
        cell = "'" + cell
        return cell[:_MAX_CSV_CELL_LEN]

    # Strip leading whitespace and U+FEFF to find significant char
    stripped = cell.lstrip("\ufeff")
    stripped = stripped.lstrip()
    if not stripped:
        return cell[:_MAX_CSV_CELL_LEN]

    if stripped[0] in _FORMULA_TRIGGERS:
        cell = "'" + cell

    return cell[:_MAX_CSV_CELL_LEN]


# ─── Markdown URI validation ──────────────────────────────────────────────
# Characters that browsers ignore inside URI schemes
_SCHEME_IGNORED_CHARS = set(
    "\t\n\r\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
    "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f"
)

# Known safe schemes — if URI starts with one literally, immediately safe.
_SAFE_SCHEMES = {"http", "https", "mailto"}

# Maximum length of the scheme-candidate portion to decode (bytes before colon)
_MAX_SCHEME_CANDIDATE_LEN = 128


def _extract_scheme_portion(uri: str) -> tuple[str, bool]:
    """Extract bounded scheme-candidate. Returns (candidate, was_truncated)."""
    colon_idx = uri.find(":")
    if 0 < colon_idx <= _MAX_SCHEME_CANDIDATE_LEN:
        return (uri[:colon_idx + 1], False)
    truncated = len(uri) > _MAX_SCHEME_CANDIDATE_LEN
    return (uri[:_MAX_SCHEME_CANDIDATE_LEN], truncated)


def _is_dangerous_uri(uri: str) -> bool:
    """Classify a URI. Only the scheme portion is normalized."""
    if not uri:
        return False
    stripped = uri.lstrip(" \t\r\n\x0b\x0c")
    # Literal safe-scheme check
    lower_start = stripped[:10].lower()
    for safe in _SAFE_SCHEMES:
        if lower_start.startswith(safe + ":"):
            return False
    # Relative/fragment check
    if stripped and stripped[0] in (".", "/", "#"):
        return False
    # Extract scheme candidate
    candidate, was_truncated = _extract_scheme_portion(stripped)
    # Iteratively decode only the scheme candidate
    prev = None
    for _ in range(_URI_NORMALIZATION_MAX_ITER):
        if candidate == prev:
            # Stable
            if was_truncated and ":" not in candidate:
                return True  # Truncated without colon → fail closed
            return _scheme_is_dangerous(candidate)
        prev = candidate
        candidate = candidate.lstrip(" \t\r\n\x0b\x0c")
        candidate = html.unescape(candidate)
        try:
            candidate = urllib.parse.unquote(candidate)
        except Exception:
            pass
        # Narrow to scheme boundary once colon visible
        new_colon = candidate.find(":")
        if new_colon > 0:
            candidate = candidate[:new_colon + 1]
            was_truncated = False
        # Check safe scheme
        cl = candidate[:10].lower()
        for safe in _SAFE_SCHEMES:
            if cl.startswith(safe + ":"):
                return False
    # Did not stabilize → fail closed
    return True


def _scheme_is_dangerous(candidate: str) -> bool:
    """Check if a normalized candidate starts with a dangerous scheme.

    Removes browser-ignored characters from the portion before the first
    colon, then compares case-insensitively against the denylist.
    """
    colon_idx = candidate.find(":")
    if colon_idx < 0 or colon_idx > 40:
        return False
    raw_scheme = candidate[:colon_idx]
    scheme = "".join(ch for ch in raw_scheme if ch not in _SCHEME_IGNORED_CHARS)
    return scheme.lower() in _DANGEROUS_SCHEMES


def _sanitize_md_link(match: re.Match) -> str:
    """Replace dangerous inline Markdown link/image destinations."""
    bracket_part = match.group(1)
    uri = match.group(2)
    if _is_dangerous_uri(uri):
        if bracket_part.startswith("!"):
            text = bracket_part[2:-1]
        else:
            text = bracket_part[1:-1]
        return text
    return match.group(0)


def _sanitize_ref_def(match: re.Match) -> str:
    """Sanitize a reference-style link definition."""
    prefix = match.group(1)  # [id]:
    destination = match.group(2)
    rest = match.group(3) or ""
    if _is_dangerous_uri(destination):
        return prefix + "#" + rest
    return match.group(0)


def _sanitize_markdown(text: str) -> str:
    """Strip dangerous constructs from Markdown.

    Order:
    1. Active HTML elements with content
    2. Self-closing active elements
    3. Unclosed active elements
    4. HTML comments
    5. Remaining HTML tags
    6. Dangerous inline link/image destinations
    7. Dangerous reference-style link definitions
    """
    text = _ACTIVE_ELEMENT_RE.sub("", text)
    text = _SELF_CLOSING_ACTIVE_RE.sub("", text)
    text = _UNCLOSED_ACTIVE_RE.sub("", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _MD_LINK_RE.sub(_sanitize_md_link, text)
    text = _MD_REF_DEF_RE.sub(_sanitize_ref_def, text)
    return text


# ─── Format extractors ────────────────────────────────────────────────────

def extract_pdf(file_path: Path) -> str:
    """Extract page text only from a PDF."""
    try:
        import pdfplumber
    except ImportError:
        raise ExtractionError("pdfplumber not available")
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) > _MAX_PAGES:
                raise ExtractionError(f"PDF exceeds {_MAX_PAGES} page limit")
            pages_text = []
            total_size = 0
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text = _strip_control_chars(text)
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
    """Extract body paragraph text only from a DOCX."""
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
                text = _strip_control_chars(text)
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
    """Extract text with encoding detection and control-char removal."""
    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > _MAX_EXTRACTED_BYTES:
            raise ExtractionError("File exceeds extraction size limit")
        text = _normalize_encoding(raw_bytes)
        text = _strip_control_chars(text)
        return _enforce_size_limit(text)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"TXT extraction failed: {e}")


def extract_csv(file_path: Path, max_bytes: int | None = None) -> str:
    """Extract CSV with proper serialization and formula neutralization.

    Size limit enforced per-row in UTF-8 bytes (not character count).
    Rejects (ExtractionError) when limit exceeded — does not truncate mid-row.
    """
    limit = max_bytes if max_bytes is not None else _MAX_EXTRACTED_BYTES
    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > _MAX_EXTRACTED_BYTES:
            raise ExtractionError("File exceeds extraction size limit")
        text = _normalize_encoding(raw_bytes)
        text = _strip_control_chars(text)

        reader = csv.reader(io.StringIO(text))
        rows_output: list[bytes] = []
        total_bytes = 0
        row_count = 0

        for row in reader:
            row_count += 1
            if row_count > _MAX_CSV_ROWS:
                raise ExtractionError(
                    f"CSV exceeds maximum row count of {_MAX_CSV_ROWS}"
                )
            safe_row = [_neutralize_csv_cell(cell) for cell in row]
            # Serialize this single row to get its exact UTF-8 byte size
            row_buf = io.StringIO()
            row_writer = csv.writer(row_buf, lineterminator="\r\n")
            row_writer.writerow(safe_row)
            row_str = row_buf.getvalue()
            row_bytes = row_str.encode("utf-8")
            total_bytes += len(row_bytes)
            if total_bytes > limit:
                raise ExtractionError("Extracted CSV text exceeds size limit")
            rows_output.append(row_bytes)

        result_bytes = b"".join(rows_output)
        result = result_bytes.decode("utf-8")
        # Remove trailing CRLF
        if result.endswith("\r\n"):
            result = result[:-2]
        return result
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"CSV extraction failed: {e}")


def extract_md(file_path: Path) -> str:
    """Extract Markdown with sanitation and size limit."""
    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > _MAX_EXTRACTED_BYTES:
            raise ExtractionError("File exceeds extraction size limit")
        text = _normalize_encoding(raw_bytes)
        text = _strip_control_chars(text)
        text = _sanitize_markdown(text)
        return _enforce_size_limit(text)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"MD extraction failed: {e}")


def extract_content(file_path: Path, extension: str) -> str:
    """Dispatch to format-specific extractor."""
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
