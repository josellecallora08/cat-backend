"""PDF structural security validation using pypdf.

Rejects: encryption, attachments, JavaScript, active content, external links,
embedded objects, malformed structure. Performs inspection BEFORE text extraction.

Security hardening:
- One recursive function handles IndirectObject, DictionaryObject, ArrayObject
- Arrays nested inside arrays are fully inspected
- Depth and object-count limits apply to ALL object types
- Dereference failures are fail-closed (PDF_MALFORMED)
- Cycle detection via object id
"""

import io
import logging

from app.services.upload_validator import UploadRejectionReason


logger = logging.getLogger(__name__)

# Dangerous PDF name objects indicating active/unsafe content
_ACTIVE_CONTENT_KEYS = {"/JavaScript", "/JS", "/OpenAction", "/AA"}
_EMBEDDED_KEYS = {"/EmbeddedFiles", "/EmbeddedFile", "/Filespec", "/EF"}
_MEDIA_KEYS = {"/RichMedia", "/3D", "/Movie", "/Sound", "/Launch"}
_EXTERNAL_ACTION_KEYS = {"/URI", "/GoToR", "/SubmitForm", "/ImportData"}

# Max pages to inspect (resource limit)
_MAX_PAGES_INSPECT = 2000
_MAX_DEPTH = 50
_MAX_OBJECTS = 10000


def validate_pdf_security(
    file_bytes: bytes,
) -> tuple[bool, UploadRejectionReason | None]:
    """Validate PDF structural security before extraction.

    Checks for encryption, attachments, active content, embedded objects,
    external links, and malformed structure using pypdf.

    Returns:
        (True, None) if safe, (False, reason) if rejected.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not available for PDF security validation")
        return (False, UploadRejectionReason.PDF_UNSAFE_STRUCTURE)

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception:
        return (False, UploadRejectionReason.PDF_MALFORMED)

    # 1. Encryption check
    if reader.is_encrypted:
        return (False, UploadRejectionReason.PDF_ENCRYPTED)

    # 2. Inspect all objects in the PDF for unsafe structures
    try:
        result = _inspect_pdf_objects(reader)
        if result is not None:
            return (False, result)
    except RecursionError:
        return (False, UploadRejectionReason.PDF_UNSAFE_STRUCTURE)
    except Exception as e:
        logger.warning("PDF inspection error (fail-closed): %s", e)
        return (False, UploadRejectionReason.PDF_MALFORMED)

    # 3. Resource limit: page count
    try:
        if len(reader.pages) > _MAX_PAGES_INSPECT:
            return (False, UploadRejectionReason.PDF_UNSAFE_STRUCTURE)
    except Exception:
        return (False, UploadRejectionReason.PDF_MALFORMED)

    return (True, None)


def _inspect_pdf_objects(reader) -> UploadRejectionReason | None:
    """Walk PDF object tree looking for unsafe structures.

    One unified recursive function handles:
    - IndirectObject (dereference and recurse)
    - DictionaryObject (check keys, recurse values)
    - ArrayObject (recurse each element, including nested arrays)

    Fail-closed: any traversal/access error returns PDF_MALFORMED.
    Depth and object-count limits apply to all object types.
    """
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        IndirectObject,
    )

    visited = set()
    object_count = [0]

    def _check_key(key_str: str) -> UploadRejectionReason | None:
        """Check if a dictionary key indicates unsafe content."""
        if key_str in _EMBEDDED_KEYS:
            return UploadRejectionReason.PDF_ATTACHMENT
        if key_str in _ACTIVE_CONTENT_KEYS:
            return UploadRejectionReason.PDF_ACTIVE_CONTENT
        if key_str in _MEDIA_KEYS:
            return UploadRejectionReason.PDF_EMBEDDED_OBJECT
        if key_str in _EXTERNAL_ACTION_KEYS:
            return UploadRejectionReason.PDF_EXTERNAL_LINK
        return None

    def _inspect(obj, depth: int = 0) -> UploadRejectionReason | None:
        """Recursively inspect any PDF object. Handles Dict, Array, Indirect."""
        if depth > _MAX_DEPTH:
            return UploadRejectionReason.PDF_UNSAFE_STRUCTURE

        object_count[0] += 1
        if object_count[0] > _MAX_OBJECTS:
            return UploadRejectionReason.PDF_UNSAFE_STRUCTURE

        # Cycle detection
        obj_id = id(obj)
        if obj_id in visited:
            return None
        visited.add(obj_id)

        # Dereference IndirectObject
        if isinstance(obj, IndirectObject):
            try:
                obj = obj.get_object()
            except Exception:
                return UploadRejectionReason.PDF_MALFORMED
            # If dereference returns None or NullObject, fail closed
            if obj is None:
                return UploadRejectionReason.PDF_MALFORMED
            from pypdf.generic import NullObject
            if isinstance(obj, NullObject):
                return UploadRejectionReason.PDF_MALFORMED
            # After dereference, re-check id for cycles
            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)

        # Handle DictionaryObject
        if isinstance(obj, DictionaryObject):
            for key in obj.keys():
                key_str = str(key)

                # Check dangerous keys
                result = _check_key(key_str)
                if result is not None:
                    return result

                # Check /Subtype for FileAttachment
                if key_str == "/Subtype":
                    try:
                        val = str(obj.get(key, ""))
                        if val == "/FileAttachment":
                            return UploadRejectionReason.PDF_ATTACHMENT
                    except Exception:
                        return UploadRejectionReason.PDF_MALFORMED

                # Recurse into value
                try:
                    value = obj[key]
                except Exception:
                    return UploadRejectionReason.PDF_MALFORMED

                result = _inspect(value, depth + 1)
                if result is not None:
                    return result

        # Handle ArrayObject - recurse into EVERY element
        elif isinstance(obj, ArrayObject):
            for item in obj:
                result = _inspect(item, depth + 1)
                if result is not None:
                    return result

        return None

    # Check each page
    try:
        for page in reader.pages[:_MAX_PAGES_INSPECT]:
            try:
                page_obj = page.get_object() if hasattr(page, "get_object") else page
            except Exception:
                return UploadRejectionReason.PDF_MALFORMED
            if isinstance(page_obj, DictionaryObject):
                result = _inspect(page_obj)
                if result:
                    return result
    except Exception:
        return UploadRejectionReason.PDF_MALFORMED

    # Check the document catalog
    try:
        if hasattr(reader, "_root_object"):
            root = reader._root_object
        elif hasattr(reader, "root_object"):
            root = reader.root_object
        else:
            root = reader.trailer.get("/Root")
            if root and hasattr(root, "get_object"):
                root = root.get_object()

        if root and isinstance(root, DictionaryObject):
            result = _inspect(root)
            if result:
                return result
    except Exception:
        return UploadRejectionReason.PDF_MALFORMED

    return None
