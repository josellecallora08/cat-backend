"""Tests for safe report failure records."""

import pytest

from app.services.report_failures import (
    FailureClass,
    build_failure_context,
    classify_failure,
    redact_sensitive,
    safe_message_for_status,
)


def test_failure_context_has_safe_status_message_and_opaque_correlation_id() -> None:
    failure = build_failure_context(
        component="report",
        method="get",
        route="/api/sessions/abc/report",
        status=500,
        detail="database password=do-not-expose",
    )

    assert failure.failure_class is FailureClass.BACKEND
    assert failure.safe_message == "The report is temporarily unavailable. Please try again later."
    assert failure.correlation_id
    assert "password" not in failure.safe_message.lower()
    assert "do-not-expose" not in failure.to_dict().__repr__()


def test_redaction_removes_credentials_pii_sql_and_internal_paths() -> None:
    value = (
        "Bearer testkey password=testkey api_key=testkey "
        "alice@example.com +1 (555) 123-4567 "
        "SELECT password FROM users WHERE email='alice@example.com' "
        "at /Users/soj/CAT/app.py"
    )

    redacted = redact_sensitive(value)

    assert "jwt-value" not in redacted
    assert "secret" not in redacted
    assert "key-123" not in redacted
    assert "alice@example.com" not in redacted
    assert "/Users/soj/CAT" not in redacted
    assert "SELECT password FROM" not in redacted


def test_status_messages_cover_supported_http_failure_classes() -> None:
    assert safe_message_for_status(400) == "The report request was invalid."
    assert safe_message_for_status(401) == "You must sign in to view this report."
    assert safe_message_for_status(403) == "You are not allowed to view this report."
    assert safe_message_for_status(404) == "The requested report was not found."
    assert safe_message_for_status(422) == "The report data could not be validated."
    assert safe_message_for_status(429) == "The report request could not be completed."
    assert (
        safe_message_for_status(503)
        == "The report is temporarily unavailable. Please try again later."
    )
    assert classify_failure(status=422) is FailureClass.DATA_CONTRACT
    assert classify_failure(failure_class="export", status=500) is FailureClass.EXPORT


def test_failure_context_redacts_detail_and_preserves_safe_context() -> None:
    failure = build_failure_context(
        component="backend",
        method="get",
        route="/api/sessions/1/report",
        status=500,
        detail="SELECT password FROM users WHERE email='alice@example.com'",
        assertion_context='File "/Users/soj/CAT/app.py", line 10',
        code="database_failure",
    )

    record = failure.to_dict()
    assert record["component"] == "backend"
    assert record["method"] == "GET"
    assert record["route"] == "/api/sessions/1/report"
    assert record["code"] == "database_failure"
    assert "alice@example.com" not in repr(record)
    assert "SELECT password FROM" not in repr(record)
    assert "/Users/soj/CAT" not in repr(record)


def test_context_redacts_assertion_details_and_normalizes_request_fields() -> None:
    failure = build_failure_context(
        component="http_e2e",
        method="post",
        route="/api/report?token=testkey",
        status=422,
        assertion_context="Authorization: Bearer abc; email alice@example.com",
    )

    assert failure.failure_class is FailureClass.HTTP_E2E
    assert failure.method == "POST"
    assert "testkey" not in failure.route
    assert "abc" not in (failure.assertion_context or "")
    assert "alice@example.com" not in (failure.assertion_context or "")


@pytest.mark.parametrize("failure_class", list(FailureClass))
def test_classification_maps_every_failure_class(failure_class: FailureClass) -> None:
    """Every release-gate category is preserved in a failure record."""
    failure = build_failure_context(
        component="report",
        method="GET",
        route="/api/sessions/session-id/report",
        status=500,
        failure_class=failure_class,
        assertion_context="expected=200 actual=500",
    )

    record = failure.to_dict()
    assert record["failure_class"] == failure_class.value
    assert record["method"] == "GET"
    assert record["route"] == "/api/sessions/session-id/report"
    assert record["status"] == 500
    assert record["assertion_context"] == "expected=200 actual=500"


def test_http_failure_record_keeps_http_context_and_redacts_sensitive_payload() -> None:
    """HTTP release-gate records retain diagnostics without leaking request data."""
    sensitive = (
        "Authorization: Bearer jwt-secret token=access-secret credential=credential-secret "
        "alice@example.com +1 (555) 123-4567 "
        "SELECT password FROM users WHERE email='alice@example.com' "
        'File "/Users/soj/CAT/app.py", line 42, in handle_request'
    )
    failure = build_failure_context(
        component="http_e2e",
        method="post",
        route="/api/sessions/session-id/report?token=query-secret",
        status=422,
        failure_class=FailureClass.HTTP_E2E,
        detail=sensitive,
        assertion_context=sensitive,
        code="contract_validation",
    )

    record = repr(failure.to_dict())
    assert failure.failure_class is FailureClass.HTTP_E2E
    assert failure.method == "POST"
    assert failure.status == 422
    assert failure.route.startswith("/api/sessions/session-id/report")
    assert "jwt-secret" not in record
    assert "access-secret" not in record
    assert "credential-secret" not in record
    assert "alice@example.com" not in record
    assert "+1 (555) 123-4567" not in record
    assert "SELECT password FROM" not in record
    assert "/Users/soj/CAT/app.py" not in record
    assert "query-secret" not in record
