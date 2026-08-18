"""Property tests for safe report failure presentation."""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.report_failures import FailureClass, build_failure_context


_SAFE_FAILURE_CLASSES = st.sampled_from(list(FailureClass))
_SAFE_STATUSES = st.sampled_from([400, 401, 403, 404, 422, 429, 500, 503])
_SAFE_TOKENS = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
    min_size=8,
    max_size=24,
)


@settings(max_examples=100)
@given(
    failure_class=_SAFE_FAILURE_CLASSES,
    status=_SAFE_STATUSES,
    token=_SAFE_TOKENS,
)
def test_safe_failure_presentation_never_exposes_sensitive_implementation_data(
    failure_class: FailureClass,
    status: int,
    token: str,
) -> None:
    """Feature: report-quality-release-gates, Property 8.

    **Validates: Requirements 2.2-2.3, 8.5, 10.5, 10.7**
    """
    email = f"user{token}@example.test"
    sql_text = "SELECT password FROM users WHERE email='user@example.test'"
    sensitive_detail = (
        f"Bearer {token} password={token} api_key={token} {email} {sql_text} "
        f"at /Users/private/{token}/app.py, line 42\n"
    )

    failure = build_failure_context(
        component=failure_class.value,
        method="post",
        route=f"/api/sessions/report?token={token}",
        status=status,
        failure_class=failure_class,
        detail=sensitive_detail,
        assertion_context=sensitive_detail,
    )
    presentation = repr(failure.to_dict())

    assert token not in presentation
    assert email not in presentation
    assert "SELECT password FROM" not in presentation
    assert "/Users/private/" not in presentation
    assert f"Bearer {token}" not in presentation
    assert failure.failure_class is failure_class
    assert failure.safe_message
    assert failure.correlation_id
