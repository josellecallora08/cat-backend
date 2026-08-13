"""DOCX/OOXML security validation beyond archive-level checks.

Inspects relationship files for external targets, OLE/ActiveX embeddings,
DTD declarations, and unsafe XML structures. Called AFTER validate_docx_archive()
passes (which handles encryption, macros, zip bombs, path traversal).

Security hardening:
- Rejects ALL entries under word/embeddings/ and word/activeX/ regardless of extension
- Case-insensitive part discovery for .rels and .xml files
- Fail-closed on any read error (never silently accept unreadable parts)
- Uses defusedxml for hardened parsing (forbids DTD, entities, external refs)
- Handles all XML encodings (UTF-8/16 LE/BE, with/without BOM)
- Parser unavailability fails closed
"""

import logging
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element

from app.services.upload_validator import UploadRejectionReason


logger = logging.getLogger(__name__)

# Relationship types that indicate unsafe content
_UNSAFE_REL_TYPES = {
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject",
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/package",
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk",
    "http://schemas.microsoft.com/office/2006/relationships/activeXControl",
    "http://schemas.microsoft.com/office/2006/relationships/activeXControlBinary",
}

# Normalized path prefixes that are ALWAYS rejected (case-insensitive).
# ANY file under these directories is rejected regardless of extension.
_UNSAFE_DIR_PREFIXES = ("word/embeddings/", "word/activex/")


def _safe_parse_xml(raw_bytes: bytes) -> Element | UploadRejectionReason:
    """Parse XML bytes using defusedxml with all dangerous features disabled.

    Handles any encoding declared by the XML document (UTF-8, UTF-16 LE/BE,
    with or without BOM). Rejects DTD, entities, external references.

    Returns:
        Element root on success, or UploadRejectionReason on failure.
    """
    try:
        import defusedxml.ElementTree as SafeET
    except ImportError:
        # defusedxml unavailable — fail closed
        logger.error("defusedxml not available; rejecting XML part (fail-closed)")
        return UploadRejectionReason.DOCX_UNSAFE_XML

    try:
        # defusedxml.fromstring handles encoding via XML declaration.
        # Explicitly forbid DTD, entities, and external references.
        # Pass raw bytes so the XML parser respects the encoding declaration.
        root = SafeET.fromstring(
            raw_bytes,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
        return root
    except SafeET.DTDForbidden:
        return UploadRejectionReason.DOCX_UNSAFE_XML
    except SafeET.EntitiesForbidden:
        return UploadRejectionReason.DOCX_UNSAFE_XML
    except SafeET.ExternalReferenceForbidden:
        return UploadRejectionReason.DOCX_UNSAFE_XML
    except Exception:
        # Malformed XML, encoding errors, etc. — fail closed
        return UploadRejectionReason.DOCX_UNSAFE_XML


def validate_docx_security(
    file_path: Path,
) -> tuple[bool, UploadRejectionReason | None]:
    """Validate DOCX content security after archive-level checks pass.

    Checks for:
    - ALL entries under word/embeddings/ (any extension)
    - ALL entries under word/activeX/ (case-insensitive, any extension)
    - External relationships (TargetMode="External", case-insensitive)
    - Unsafe OLE/object relationship types
    - DTD/entity declarations in XML files (any encoding)
    - Case-insensitive .rels and .xml part discovery
    - Fail-closed on unreadable parts
    - Hardened XML parsing via defusedxml

    Normal embedded images in word/media/ are allowed.

    Returns:
        (True, None) if safe, (False, reason) if rejected.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            entries = zf.namelist()

            # Check for unsafe path prefixes (case-insensitive, any extension)
            for entry in entries:
                entry_normalized = entry.replace("\\", "/").lower()
                for unsafe_prefix in _UNSAFE_DIR_PREFIXES:
                    if entry_normalized.startswith(unsafe_prefix):
                        return (False, UploadRejectionReason.DOCX_EMBEDDED_OBJECT)

            # Inspect all .rels files (case-insensitive discovery)
            rels_files = [
                e for e in entries if e.lower().endswith(".rels")
            ]
            for rels_file in rels_files:
                result = _check_relationships(zf, rels_file)
                if result is not None:
                    return (False, result)

            # Check XML files for safety (case-insensitive discovery)
            # Skip files already checked as .rels (they end with .rels not .xml)
            xml_files = [
                e for e in entries
                if e.lower().endswith(".xml")
            ]
            for xml_file in xml_files:
                result = _check_xml_safety(zf, xml_file)
                if result is not None:
                    return (False, result)

    except zipfile.BadZipFile:
        return (False, UploadRejectionReason.DOCX_MALFORMED)
    except Exception as e:
        logger.warning("DOCX security check failed (fail-closed): %s", e)
        return (False, UploadRejectionReason.DOCX_MALFORMED)

    return (True, None)


def _check_relationships(
    zf: zipfile.ZipFile, rels_path: str
) -> UploadRejectionReason | None:
    """Inspect a .rels file for external targets and unsafe relationship types.

    Uses hardened XML parsing. Fail-closed on read or parse errors.
    Rejects ANY relationship with TargetMode="External" (case-insensitive).
    """
    try:
        raw = zf.read(rels_path)
    except Exception:
        # Fail closed: cannot read a listed part
        return UploadRejectionReason.DOCX_MALFORMED

    # Parse with hardened parser (handles encoding, forbids DTD/entities)
    parse_result = _safe_parse_xml(raw)
    if isinstance(parse_result, UploadRejectionReason):
        return parse_result

    root = parse_result

    # Check ALL elements that look like Relationship (with or without namespace)
    for rel in root.iter():
        if not rel.tag.endswith("Relationship"):
            continue

        rel_type = rel.get("Type", "")
        target_mode = rel.get("TargetMode", "")

        # Reject unsafe relationship types (OLE, ActiveX, package)
        if rel_type in _UNSAFE_REL_TYPES:
            return UploadRejectionReason.DOCX_EMBEDDED_OBJECT

        # Reject ANY external relationship regardless of target URI
        if target_mode.lower() == "external":
            return UploadRejectionReason.DOCX_EXTERNAL_LINK

    return None


def _check_xml_safety(
    zf: zipfile.ZipFile, xml_path: str
) -> UploadRejectionReason | None:
    """Check XML file for safety using hardened parser.

    Handles all encodings. Rejects DTD, entities, malformed XML.
    Fail-closed on read errors.
    """
    try:
        raw = zf.read(xml_path)
    except Exception:
        # Fail closed: cannot read a listed XML part
        return UploadRejectionReason.DOCX_UNSAFE_XML

    # Parse with hardened parser — rejects DTD/entities/external refs
    parse_result = _safe_parse_xml(raw)
    if isinstance(parse_result, UploadRejectionReason):
        return parse_result

    # XML is structurally safe
    return None
