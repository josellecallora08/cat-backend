"""Audit logging for security-sensitive operations.

Logs structured events without exposing secrets (tokens, passwords, URLs).
"""

import logging
from datetime import datetime, timezone
from uuid import UUID


audit_logger = logging.getLogger("cats.audit")


def log_reset_requested(email: str, ip: str, user_found: bool) -> None:
    """Log a password reset request."""
    audit_logger.info(
        "RESET_REQUESTED email=%s ip=%s user_found=%s",
        _mask_email(email),
        ip,
        user_found,
    )


def log_reset_success(user_id: str, ip: str) -> None:
    """Log a successful password reset."""
    audit_logger.info(
        "RESET_SUCCESS user_id=%s ip=%s",
        user_id,
        ip,
    )


def log_reset_invalid_token(ip: str, reason: str) -> None:
    """Log an invalid/expired reset token attempt."""
    audit_logger.warning(
        "RESET_INVALID_TOKEN ip=%s reason=%s",
        ip,
        reason,
    )


def log_reset_rate_limited(ip: str, key_type: str) -> None:
    """Log a rate-limited reset attempt."""
    audit_logger.warning(
        "RESET_RATE_LIMITED ip=%s key_type=%s",
        ip,
        key_type,
    )


def log_reset_weak_password(user_id: str, ip: str, reason: str) -> None:
    """Log a password reset rejection due to weak password."""
    audit_logger.info(
        "RESET_WEAK_PASSWORD user_id=%s ip=%s reason=%s",
        user_id,
        ip,
        reason,
    )


def log_session_report_generated(
    session_id: str, report_version: int, generated_by: str | None
) -> None:
    """Log successful session report generation."""
    audit_logger.info(
        "SESSION_REPORT_GENERATED session_id=%s report_version=%s generated_by=%s",
        session_id,
        report_version,
        generated_by or "system",
    )


_SAFE_REPORT_FAILURE_CODES = frozenset(
    {
        "generation_failed",
        "artifact_missing",
        "empty_transcript",
        "not_applicable",
        "session_too_short",
        "legacy_only",
        "no_evidence",
        "no_coaching",
        "no_learning_plan",
    }
)


def log_session_report_generation_failed(session_id: str, report_version: int, reason: str) -> None:
    """Log only a finite safe failure classification for report generation."""
    safe_reason = reason if reason in _SAFE_REPORT_FAILURE_CODES else "generation_failed"
    audit_logger.warning(
        "SESSION_REPORT_GENERATION_FAILED session_id=%s report_version=%s reason=%s",
        session_id,
        report_version,
        safe_reason,
    )


def _mask_email(email: str) -> str:
    """Mask email for logging: show first 2 chars + domain."""
    try:
        local, domain = email.split("@", 1)
        masked_local = local[0] + "***" if len(local) <= 2 else local[:2] + "***"
        return f"{masked_local}@{domain}"
    except (ValueError, IndexError):
        return "***@***"


def log_rubric_seed_completed(
    run_id: UUID,
    source_id: str,
    initiated_by: UUID | None,
    status: str,
    reused_rubrics: int,
    created_rubrics: int,
    reused_versions: int,
    created_versions: int,
    published_versions: int,
    rejected_definitions: int,
) -> None:
    """Log a safe summary of a rubric seed run without source contents or secrets."""
    audit_logger.info(
        "RUBRIC_SEED_COMPLETED",
        extra={
            "event": "rubric_seed_completed",
            "run_id": str(run_id),
            "source_id": source_id,
            "initiated_by": str(initiated_by) if initiated_by else "system",
            "status": status,
            "reused_rubrics": reused_rubrics,
            "created_rubrics": created_rubrics,
            "reused_versions": reused_versions,
            "created_versions": created_versions,
            "published_versions": published_versions,
            "rejected_definitions": rejected_definitions,
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.11 compatibility
        },
    )
