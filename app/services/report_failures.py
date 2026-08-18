"""Safe failure classification and redaction for report APIs and release checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from secrets import token_urlsafe
from typing import Any


class FailureClass(StrEnum):
    """Stable failure categories shared by report responses and release artifacts."""

    BACKEND = "backend"
    FRONTEND = "frontend"
    HTTP_E2E = "http_e2e"
    DATA_CONTRACT = "data_contract"
    ACCESSIBILITY = "accessibility"
    RESPONSIVE_LAYOUT = "responsive_layout"
    EXPORT = "export"
    PRINT = "print"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True, slots=True)
class FailureContext:
    """Non-sensitive context describing a report or release-gate failure."""

    failure_class: FailureClass
    component: str
    method: str
    route: str
    status: int | None
    safe_message: str
    correlation_id: str
    assertion_context: str | None = None
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible failure record."""
        result = asdict(self)
        result["failure_class"] = self.failure_class.value
        return result


_SENSITIVE_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)((?:token|secret|password|passwd|credential|api[_ -]?key|"
            r"authorization)\s*[:=]\s*)[^\s,;]+"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)(?:postgres(?:ql)?|mysql|sqlite(?:3)?|mongodb)://[^\s'\"]+"),
        "[REDACTED DATABASE URL]",
    ),
    (
        re.compile(
            r"(?i)\b(?:select|insert|update|delete|drop|alter|create)\b[\s\S]{0,300}?\b(?:from|into|table|where)\b[\s\S]{0,300}"
        ),
        "[REDACTED SQL]",
    ),
    (re.compile(r"(?m)^\s*File \"[^\"]+\", line \d+[^\n]*\n?"), "[REDACTED STACK FRAME]"),
    (
        re.compile(r"(?i)(?:/Users/|/home/|/workspace/|[A-Z]:\\)[^\s'\"]+"),
        "[REDACTED INTERNAL PATH]",
    ),
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "[REDACTED EMAIL]"),
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"), "[REDACTED PHONE]"),
)

_SAFE_MESSAGES = {
    400: "The report request was invalid.",
    401: "You must sign in to view this report.",
    403: "You are not allowed to view this report.",
    404: "The requested report was not found.",
    422: "The report data could not be validated.",
}


def new_correlation_id() -> str:
    """Create an opaque, URL-safe identifier without embedding request data."""
    return token_urlsafe(16)


def redact_sensitive(value: Any, *, max_length: int = 500) -> str:
    """Remove credentials, PII, SQL, stack frames, and internal paths from text."""
    text = str(value) if value is not None else ""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:max_length].strip()


def safe_message_for_status(status: int | None) -> str:
    """Return a user-safe message for an HTTP status without exposing internals."""
    if status in _SAFE_MESSAGES:
        return _SAFE_MESSAGES[status]
    if status is not None and 400 <= status < 500:
        return "The report request could not be completed."
    return "The report is temporarily unavailable. Please try again later."


def classify_failure(
    *,
    status: int | None = None,
    failure_class: FailureClass | str | None = None,
    component: str = "report",
) -> FailureClass:
    """Choose a stable failure class, using component and status as safe defaults."""
    if failure_class is not None:
        return FailureClass(failure_class)
    if component in {item.value for item in FailureClass}:
        return FailureClass(component)
    if status in {400, 422}:
        return FailureClass.DATA_CONTRACT
    return FailureClass.BACKEND


def build_failure_context(
    *,
    component: str,
    method: str,
    route: str,
    status: int | None = None,
    failure_class: FailureClass | str | None = None,
    detail: Any = None,
    assertion_context: Any = None,
    code: str | None = None,
    correlation_id: str | None = None,
) -> FailureContext:
    """Build a redacted failure record suitable for JSON responses or artifacts."""
    selected_class = classify_failure(
        status=status, failure_class=failure_class, component=component
    )
    return FailureContext(
        failure_class=selected_class,
        component=redact_sensitive(component, max_length=80),
        method=redact_sensitive(method.upper(), max_length=12),
        route=redact_sensitive(route, max_length=200),
        status=status,
        safe_message=safe_message_for_status(status),
        correlation_id=correlation_id or new_correlation_id(),
        assertion_context=redact_sensitive(assertion_context) if assertion_context else None,
        code=redact_sensitive(code, max_length=80) if code else None,
    )
