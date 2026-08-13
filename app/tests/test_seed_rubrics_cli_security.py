"""Focused operational and authorization tests for rubric seeding."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi import HTTPException
from scripts import seed_rubrics as seed_module

from app.config import settings
from app.models.user import User, UserRole
from app.services.auth import require_admin, require_auth


class ExplodingSession:
    """Fail if disabled or source configuration paths touch the database."""

    def begin(self):
        raise AssertionError("database mutation was attempted")


@pytest.mark.asyncio
async def test_disabled_seed_skips_database_and_returns_safe_result(monkeypatch) -> None:
    """Disabled configuration reports disabled without reading or changing data."""
    monkeypatch.setattr(settings, "rubric_seed_enabled", False)

    result = await seed_module.seed_rubrics(ExplodingSession())

    assert result.status == "disabled"
    assert result.source_id == "disabled"
    assert result.definition_outcomes == []


@pytest.mark.asyncio
async def test_enabled_seed_with_missing_source_fails_without_database_change(monkeypatch) -> None:
    """Missing source configuration is classified safely before persistence."""
    monkeypatch.setattr(settings, "rubric_seed_enabled", True)
    monkeypatch.setattr(settings, "rubric_source", "")
    sensitive_marker = "database-password-token"

    result = await seed_module.seed_rubrics(ExplodingSession())

    assert result.status == "failed"
    assert result.definition_outcomes == []
    assert result.source_id == "unknown"
    assert sensitive_marker not in result.model_dump_json()
    assert "password" not in result.model_dump_json().lower()


def test_cli_emits_only_safe_json_and_returns_success(monkeypatch, capsys) -> None:
    """Successful CLI execution emits one JSON result and exits zero."""
    safe_result = seed_module.SeedRunResult.new("approved-rubrics-v1")
    monkeypatch.setattr(seed_module, "run_seed", _async_result(safe_result))

    exit_code = seed_module.main()
    output = capsys.readouterr()
    payload = json.loads(output.out)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert output.err == ""
    assert "password" not in output.out.lower()
    assert "token" not in output.out.lower()


def test_cli_returns_nonzero_for_failed_result(monkeypatch, capsys) -> None:
    """Failed seed results use a non-zero CLI exit code without traceback output."""
    failed = seed_module.SeedRunResult.new("unknown")
    failed.status = "failed"
    failed.warnings = ["Rubric source could not be read."]
    monkeypatch.setattr(seed_module, "run_seed", _async_result(failed))

    assert seed_module.main() == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "failed"
    assert output.err == ""
    assert "Traceback" not in output.out


@pytest.mark.asyncio
async def test_seed_authorization_rejects_unauthenticated_and_non_admin() -> None:
    """The existing admin dependency rejects callers before any service invocation."""
    with pytest.raises(HTTPException) as unauthenticated:
        await require_auth(None)
    assert unauthenticated.value.status_code == 401

    non_admin = User(
        id=uuid4(),
        email="agent@example.test",
        full_name="Agent",
        role=UserRole.USER.value,
        user_type="agent",
    )
    with pytest.raises(HTTPException) as forbidden:
        await require_admin(non_admin)
    assert forbidden.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_dependency_accepts_admin_principal() -> None:
    """An authenticated admin principal is the only accepted seed authorization."""
    admin = User(
        id=uuid4(),
        email="admin@example.test",
        full_name="Administrator",
        role=UserRole.ADMIN.value,
        user_type=None,
    )

    assert await require_admin(admin) is admin


def _async_result(result: seed_module.SeedRunResult):
    """Create an async callable suitable for replacing the CLI runner."""

    async def runner() -> seed_module.SeedRunResult:
        return result

    return runner


@pytest.mark.asyncio
async def test_failed_seed_result_survives_audit_failure(monkeypatch) -> None:
    """Audit delivery failures do not prevent returning a safe failed result."""
    monkeypatch.setattr(settings, "rubric_seed_enabled", True)
    monkeypatch.setattr(settings, "rubric_source", "")

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("audit backend unavailable")

    monkeypatch.setattr(seed_module, "log_rubric_seed_completed", fail_audit)

    result = await seed_module.seed_rubrics(ExplodingSession())

    assert result.status == "failed"
    assert result.source_id == "unknown"
    assert "audit backend unavailable" not in result.model_dump_json()
