"""Audit logging for security-sensitive operations.

Logs structured events without exposing secrets (tokens, passwords, URLs).
"""

import logging
from typing import Optional

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


def _mask_email(email: str) -> str:
    """Mask email for logging: show first 2 chars + domain."""
    try:
        local, domain = email.split("@", 1)
        if len(local) <= 2:
            masked_local = local[0] + "***"
        else:
            masked_local = local[:2] + "***"
        return f"{masked_local}@{domain}"
    except (ValueError, IndexError):
        return "***@***"
