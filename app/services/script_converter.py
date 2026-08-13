"""S1-09: Convert sanitized extracted content into ScriptContract fields.

Accepts sanitized text from ScriptUpload.extracted_content and produces
a ScriptContract-compatible dictionary suitable for script_registry.create_draft().

Strategy:
1. If text is valid JSON matching ScriptContract structure → use directly.
2. If text is valid YAML matching ScriptContract structure → use directly.
3. Malformed JSON/YAML that clearly attempts to be structured → raise ConversionError.
4. Plain unstructured prose → raise ConversionError requiring manual mapping.

S1-09 conversion boundary: the upload conversion endpoint accepts a complete
ScriptContract serialized as JSON or YAML. Ordinary prose extracted from a
PDF/DOCX/TXT upload is rejected for manual mapping; it must not be silently
invented into financial or behavioral values. The Script_Registry requirements
define JSON/YAML as the supported Script definition formats, and this
conversion boundary preserves that contract.

IMPORTANT: This converter does NOT fabricate financial or behavioral values.
If the input does not contain valid, complete ScriptContract data, conversion
fails with an actionable error rather than silently inventing defaults.
"""

import json
import logging
import re
from typing import Any

import yaml
from pydantic import ValidationError

from app.config import settings
from app.schemas.script import ScriptContract


logger = logging.getLogger(__name__)

_STRUCTURED_CODE_BLOCK = re.compile(
    r"```(?P<format>json|ya?ml)\s*\r?\n(?P<body>.*?)\r?\n```",
    re.IGNORECASE | re.DOTALL,
)


class ConversionError(Exception):
    """Raised when extracted content cannot be converted to a ScriptContract."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        self.details = details or {}
        super().__init__(message)


def convert_extracted_to_contract(
    extracted_text: str, *, format_hint: str | None = None
) -> dict[str, Any]:
    """Convert sanitized extracted text into a ScriptContract-compatible dict.

    Returns a validated dictionary ready for script_registry.create_draft().

    Raises:
        ConversionError: If the content is empty, oversized, malformed,
            incomplete, or plain prose that cannot produce a valid contract
            without fabricating data.
    """
    if not extracted_text or not extracted_text.strip():
        raise ConversionError("Extracted content is empty or whitespace-only")

    # Oversized check using configurable limit
    max_size = settings.script_max_definition_size_bytes
    content_size = len(extracted_text.encode("utf-8"))
    if content_size > max_size:
        raise ConversionError(
            f"Extracted content size ({content_size} bytes) exceeds the maximum "
            f"allowed size ({max_size} bytes). Content cannot be truncated silently.",
            details={"limit_bytes": max_size, "actual_bytes": content_size},
        )

    text = extracted_text.strip()

    # Markdown documents commonly wrap the authoritative ScriptContract in a
    # fenced JSON/YAML block. Parse that block rather than misclassifying the
    # document heading (for example "# Script: Name") as YAML.
    fenced = _STRUCTURED_CODE_BLOCK.search(text)
    if fenced:
        fenced_format = fenced.group("format").lower()
        fenced_text = fenced.group("body").strip()
        if fenced_format == "json":
            return _parse_json_contract(fenced_text)
        return _parse_yaml_contract(fenced_text)

    # Detect whether the content attempts to be JSON
    is_json_attempt = text.startswith("{") or text.startswith("[")
    # Detect whether the content attempts to be YAML with mapping structure
    first_meaningful_line = next(
        (line.strip() for line in text.splitlines() if line.strip()), ""
    )
    is_yaml_attempt = (
        ":" in first_meaningful_line
        and not first_meaningful_line.startswith(("#", "-", "*", ">"))
        and not is_json_attempt
    )

    # If a format_hint is provided, use it to guide parsing
    if format_hint == "json" or (format_hint is None and is_json_attempt):
        return _parse_json_contract(text)

    if format_hint == "yaml" or (format_hint is None and is_yaml_attempt):
        return _parse_yaml_contract(text)

    # Plain unstructured prose — cannot create a ScriptContract without fabrication
    raise ConversionError(
        "Content is plain unstructured text. A complete ScriptContract (JSON or YAML) "
        "is required. Manual mapping or review is needed to create a script from "
        "free-text content.",
        details={"content_type": "plain_text", "action_required": "manual_mapping"},
    )


def _parse_json_contract(text: str) -> dict[str, Any]:
    """Parse text as a JSON ScriptContract. Raises ConversionError on failure."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConversionError(
            f"Content appears to be JSON but cannot be parsed: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})",
            details={"format": "json", "parse_error": exc.msg},
        ) from exc

    if not isinstance(data, dict):
        raise ConversionError(
            "Parsed JSON must be an object/mapping, got "
            f"{type(data).__name__}",
            details={"format": "json", "parsed_type": type(data).__name__},
        )

    return _validate_contract_data(data, format_name="json")


def _parse_yaml_contract(text: str) -> dict[str, Any]:
    """Parse text as a YAML ScriptContract. Raises ConversionError on failure."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # Provide safe error info without exposing full content
        error_msg = str(exc).split("\n")[0] if str(exc) else "Invalid YAML"
        raise ConversionError(
            f"Content appears to be YAML but cannot be parsed: {error_msg}",
            details={"format": "yaml", "parse_error": error_msg},
        ) from exc

    if not isinstance(data, dict):
        raise ConversionError(
            "Parsed YAML must be a mapping, got "
            f"{type(data).__name__ if data is not None else 'null'}",
            details={"format": "yaml", "parsed_type": type(data).__name__ if data is not None else "null"},
        )

    return _validate_contract_data(data, format_name="yaml")


def _validate_contract_data(data: dict[str, Any], *, format_name: str) -> dict[str, Any]:
    """Validate parsed data against ScriptContract schema.

    Raises ConversionError with actionable field-level details on failure.
    Does NOT fabricate missing fields.
    """
    try:
        contract = ScriptContract(**data)
    except ValidationError as exc:
        # Collect actionable error info without exposing raw content
        errors = []
        for error in exc.errors():
            loc = ".".join(str(p) for p in error["loc"]) or "<root>"
            msg = error["msg"]
            error_type = error["type"]
            errors.append({"field": loc, "message": msg, "type": error_type})

        raise ConversionError(
            f"Structured {format_name.upper()} content does not satisfy the "
            f"ScriptContract schema ({len(errors)} validation error(s)). "
            "All required fields must be present with correct types.",
            details={
                "format": format_name,
                "validation_errors": errors,
                "action_required": "fix_content",
            },
        ) from exc

    return contract.model_dump(mode="json")


def detect_format(text: str) -> str:
    """Detect whether text is JSON or YAML format.

    Returns "json" or "yaml". Used to pass to script_registry.create_draft().
    """
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "yaml"
