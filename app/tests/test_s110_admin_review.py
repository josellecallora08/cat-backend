"""S1-10: Admin review API — unit and integration tests (mocked DB).

Tests cover:
- Authorization: 401/403/success
- Review retrieval: all workflow states, security filtering
- Edit: validation, concurrency, rollback
- Retry: eligibility, targets
- Rejection: reason required, idempotency
- Publish: eligibility, delegation
- Allowed actions: table-driven state tests
- Audit: structured log verification
"""
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.script_upload import UploadStatus
from app.services.auth import get_current_user, require_admin
from app.services.review_service import derive_actions, derive_review_status


# ─── Fixtures ─────────────────────────────────────────────────────────────────

VALID_CONTRACT = {
    "debtor_persona": {"name": "J", "communication_style": "C", "background": "B"},
    "financial_situation": {
        "outstanding_balance": "500.00",
        "days_past_due": 10,
        "reason_for_delinquency": "Lost job",
    },
    "opening_response": "Hello",
    "expected_replies": [{"agent_statement": "Hi", "debtor_reply": "Hey"}],
    "trigger_phrases": [{"phrase": "pay", "behavior": "agree"}],
    "emotional_state_rules": [{"trigger": "yell", "state_change": "angry"}],
    "payment_conditions": [{"condition": "full", "term": "monthly", "accepted": True}],
    "escalation_conditions": [
        {"condition": "hang up", "behavior": "end", "ends_call": True}
    ],
    "prohibited_responses": ["never"],
    "conversation_goal": {"target_outcome": "pay", "completion_condition": "done"},
}


def _mock_admin():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "admin@test.com"
    user.role = "admin"
    return user


def _mock_non_admin():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "agent@test.com"
    user.role = "user"
    return user


def _make_upload(
    *,
    upload_id=None,
    status="completed",
    scan_status="clean",
    extraction_status="completed",
    extracted_content=None,
    scenario_id=None,
    script_id=None,
    uploaded_by=None,
    rejected_at=None,
    rejected_by=None,
    rejection_reason=None,
    scan_signature=None,
    extraction_error=None,
    quarantine_expires_at=None,
):
    upload = MagicMock()
    upload.id = upload_id or uuid.uuid4()
    upload.status = status
    upload.scan_status = scan_status
    upload.extraction_status = extraction_status
    upload.extracted_content = extracted_content if extracted_content is not None else json.dumps(VALID_CONTRACT)
    upload.scenario_id = scenario_id if scenario_id is not None else uuid.uuid4()
    upload.script_id = script_id
    upload.uploaded_by = uploaded_by or uuid.uuid4()
    upload.filename_original = "test.txt"
    upload.mime_type = "text/plain"
    upload.file_size_bytes = 100
    upload.content_hash = "abc123"
    upload.storage_key = "test-uuid-key.txt"
    upload.scan_signature = scan_signature
    upload.extraction_error = extraction_error
    upload.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    upload.updated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    upload.quarantine_expires_at = quarantine_expires_at or datetime(2025, 1, 2, tzinfo=timezone.utc)
    upload.deleted_at = None
    upload.rejected_at = rejected_at
    upload.rejected_by = rejected_by
    upload.rejection_reason = rejection_reason
    return upload


def _make_script(*, script_id=None, status="draft", draft_content=None, is_deleted=False):
    script = MagicMock()
    script.id = script_id or uuid.uuid4()
    script.status = status
    script.format = "json"
    script.draft_content = draft_content if draft_content is not None else VALID_CONTRACT
    script.current_version_id = None
    script.is_deleted = is_deleted
    script.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    script.updated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return script


@pytest.fixture
def admin_user():
    return _mock_admin()


@pytest.fixture
def admin_override(admin_user):
    app.dependency_overrides[require_admin] = lambda: admin_user
    yield admin_user
    app.dependency_overrides.clear()


@pytest.fixture
async def client(admin_override):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def unauth_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─── State derivation unit tests ──────────────────────────────────────────────


class TestDeriveReviewStatus:
    """Table-driven tests for derive_review_status."""

    def test_rejected_status(self):
        upload = _make_upload(status="rejected")
        assert derive_review_status(upload) == "rejected"

    def test_infected(self):
        upload = _make_upload(scan_status="infected", status="failed")
        assert derive_review_status(upload) == "infected"

    def test_scan_failed(self):
        upload = _make_upload(scan_status="error", status="failed")
        assert derive_review_status(upload) == "scan_failed"

    def test_extraction_failed(self):
        upload = _make_upload(extraction_status="failed")
        assert derive_review_status(upload) == "extraction_failed"

    def test_processing_pending(self):
        upload = _make_upload(status="pending", scan_status="pending")
        assert derive_review_status(upload) == "processing"

    def test_deleted_is_rejected(self):
        upload = _make_upload(status="deleted")
        assert derive_review_status(upload) == "rejected"

    def test_completed_no_script(self):
        upload = _make_upload(script_id=None)
        assert derive_review_status(upload) == "ready_for_conversion"

    def test_completed_with_draft_script(self):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        assert derive_review_status(upload, script) == "ready_for_review"

    def test_completed_with_published_script(self):
        script = _make_script(status="published")
        upload = _make_upload(script_id=script.id)
        assert derive_review_status(upload, script) == "published"

    def test_completed_with_unpublished_script(self):
        script = _make_script(status="unpublished")
        upload = _make_upload(script_id=script.id)
        assert derive_review_status(upload, script) == "rejected"

    def test_completed_with_deleted_script(self):
        script = _make_script(is_deleted=True)
        upload = _make_upload(script_id=script.id)
        assert derive_review_status(upload, script) == "rejected"


# ─── Allowed actions unit tests ───────────────────────────────────────────────


class TestDeriveActions:
    """Test allowed actions per state."""

    def test_ready_for_review_all_actions(self):
        upload = _make_upload()
        actions = derive_actions("ready_for_review", upload)
        assert actions["can_edit"] is True
        assert actions["can_retry"] is True
        assert actions["can_reject"] is True
        assert actions["can_publish"] is True

    def test_ready_for_conversion_retry_reject(self):
        upload = _make_upload(script_id=None)
        actions = derive_actions("ready_for_conversion", upload)
        assert actions["can_edit"] is False
        assert actions["can_retry"] is True
        assert actions["can_reject"] is True
        assert actions["can_publish"] is False

    def test_infected_only_reject(self):
        upload = _make_upload(scan_status="infected")
        actions = derive_actions("infected", upload)
        assert actions["can_edit"] is False
        assert actions["can_retry"] is False
        assert actions["can_reject"] is True
        assert actions["can_publish"] is False

    def test_published_no_actions(self):
        upload = _make_upload()
        actions = derive_actions("published", upload)
        assert all(v is False for v in actions.values())

    def test_rejected_no_actions(self):
        upload = _make_upload()
        actions = derive_actions("rejected", upload)
        assert all(v is False for v in actions.values())

    def test_processing_no_actions(self):
        upload = _make_upload()
        actions = derive_actions("processing", upload)
        assert all(v is False for v in actions.values())


# ─── Authorization tests ──────────────────────────────────────────────────────


class TestReviewAuth:
    """Authorization enforcement on review endpoints."""

    async def test_unauthenticated_returns_401(self, unauth_client):
        uid = uuid.uuid4()
        r = await unauth_client.get(f"/api/scripts/uploads/{uid}/review")
        assert r.status_code == 401

    async def test_non_admin_returns_403(self, unauth_client):
        mock_user = _mock_non_admin()
        app.dependency_overrides[get_current_user] = lambda: mock_user
        try:
            uid = uuid.uuid4()
            r = await unauth_client.get(f"/api/scripts/uploads/{uid}/review")
            assert r.status_code == 403
        finally:
            app.dependency_overrides.clear()

    async def test_admin_gets_through(self, client, admin_override):
        """Admin auth passes — will get 404 for missing upload."""
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            uid = uuid.uuid4()
            r = await client.get(f"/api/scripts/uploads/{uid}/review")
            assert r.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db, None)


# ─── Review detail tests ──────────────────────────────────────────────────────


def _mock_db_for_review(upload, script=None):
    """Mock DB that returns upload then optionally script."""
    mock_db = AsyncMock()
    call_count = {"n": 0}

    async def mock_execute(stmt):
        call_count["n"] += 1
        result = MagicMock()
        if call_count["n"] == 1:
            result.scalar_one_or_none.return_value = upload
        elif call_count["n"] == 2:
            result.scalar_one_or_none.return_value = script
        else:
            result.scalar_one_or_none.return_value = None
        return result

    mock_db.execute = mock_execute
    return mock_db


class TestReviewDetail:
    """GET /uploads/{id}/review tests."""

    async def test_completed_upload_no_script(self, client, admin_override):
        upload = _make_upload(script_id=None)
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(upload)
        try:
            r = await client.get(f"/api/scripts/uploads/{upload.id}/review")
            assert r.status_code == 200
            data = r.json()
            assert data["review_status"] == "ready_for_conversion"
            assert data["upload_id"] == str(upload.id)
            assert data["sanitized_content"] is not None
            assert data["can_retry"] is True
            assert data["can_reject"] is True
            assert data["can_publish"] is False
        finally:
            app.dependency_overrides.pop(get_db, None)

    async def test_upload_with_draft_script(self, client, admin_override):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(upload, script)
        try:
            r = await client.get(f"/api/scripts/uploads/{upload.id}/review")
            assert r.status_code == 200
            data = r.json()
            assert data["review_status"] == "ready_for_review"
            assert data["script_contract"] is not None
            assert data["script_contract"]["opening_response"] == "Hello"
            assert data["can_edit"] is True
            assert data["can_publish"] is True
        finally:
            app.dependency_overrides.pop(get_db, None)

    async def test_infected_upload_omits_content(self, client, admin_override):
        upload = _make_upload(
            scan_status="infected", status="failed", scan_signature="Eicar-Test"
        )
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(upload)
        try:
            r = await client.get(f"/api/scripts/uploads/{upload.id}/review")
            assert r.status_code == 200
            data = r.json()
            assert data["review_status"] == "infected"
            assert data["sanitized_content"] is None
            assert data["script_contract"] is None
            assert data["scan_result"]["signature"] == "Eicar-Test"
        finally:
            app.dependency_overrides.pop(get_db, None)

    async def test_missing_upload_returns_404(self, client, admin_override):
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(None)
        try:
            r = await client.get(f"/api/scripts/uploads/{uuid.uuid4()}/review")
            assert r.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db, None)

    async def test_no_storage_key_in_response(self, client, admin_override):
        upload = _make_upload()
        # Set quarantine_expires_at in the future so the warning won't fire
        upload.quarantine_expires_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(upload)
        try:
            r = await client.get(f"/api/scripts/uploads/{upload.id}/review")
            data = r.json()
            assert "storage_key" not in data
            # Ensure no filesystem paths leak
            assert "test-uuid-key.txt" not in json.dumps(data)
        finally:
            app.dependency_overrides.pop(get_db, None)

    async def test_malformed_stored_contract_produces_warning(self, client, admin_override):
        script = _make_script(draft_content={"invalid": "data"})
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(upload, script)
        try:
            r = await client.get(f"/api/scripts/uploads/{upload.id}/review")
            assert r.status_code == 200
            data = r.json()
            assert data["script_contract"] is None
            warnings = data["warnings"]
            codes = [w["code"] for w in warnings]
            assert "content_unparseable" in codes
        finally:
            app.dependency_overrides.pop(get_db, None)

    async def test_uploaded_by_included(self, client, admin_override):
        uploader_id = uuid.uuid4()
        upload = _make_upload(uploaded_by=uploader_id)
        from app.database import get_session as get_db
        app.dependency_overrides[get_db] = lambda: _mock_db_for_review(upload)
        try:
            r = await client.get(f"/api/scripts/uploads/{upload.id}/review")
            data = r.json()
            assert data["uploaded_by"] == str(uploader_id)
        finally:
            app.dependency_overrides.pop(get_db, None)


# ─── Edit tests ───────────────────────────────────────────────────────────────


def _mock_db_for_edit(upload, script):
    """Mock DB for edit: 1st call=upload(locked), 2nd=script(locked)."""
    mock_db = AsyncMock()
    call_count = {"n": 0}

    async def mock_execute(stmt):
        call_count["n"] += 1
        result = MagicMock()
        if call_count["n"] == 1:
            result.scalar_one_or_none.return_value = upload
        elif call_count["n"] == 2:
            result.scalar_one_or_none.return_value = script
        else:
            result.scalar_one_or_none.return_value = None
        return result

    mock_db.execute = mock_execute
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()
    mock_db.rollback = AsyncMock()
    return mock_db


class TestEditReview:
    """PATCH /uploads/{id}/review tests."""

    async def test_valid_edit_succeeds(self, client, admin_override):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch(
            "app.services.script_registry.update_draft",
            new_callable=AsyncMock,
            return_value=script,
        ):
            r = await client.patch(
                f"/api/scripts/uploads/{upload.id}/review",
                json={"script_contract": VALID_CONTRACT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["review_status"] == "ready_for_review"
            assert data["script_contract"] is not None
        app.dependency_overrides.pop(get_db, None)

    async def test_missing_field_rejected(self, client, admin_override):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        bad = {"opening_response": "Hi"}  # missing fields
        r = await client.patch(
            f"/api/scripts/uploads/{upload.id}/review",
            json={"script_contract": bad},
        )
        assert r.status_code == 422
        app.dependency_overrides.pop(get_db, None)

    async def test_unknown_field_rejected(self, client, admin_override):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        bad = {**VALID_CONTRACT, "evil_field": "injection"}
        r = await client.patch(
            f"/api/scripts/uploads/{upload.id}/review",
            json={"script_contract": bad},
        )
        assert r.status_code == 422
        app.dependency_overrides.pop(get_db, None)

    async def test_no_linked_script_rejected(self, client, admin_override):
        upload = _make_upload(script_id=None)
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.patch(
            f"/api/scripts/uploads/{upload.id}/review",
            json={"script_contract": VALID_CONTRACT},
        )
        assert r.status_code == 422
        assert "no_linked_script" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_published_script_edit_rejected(self, client, admin_override):
        script = _make_script(status="published")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.patch(
            f"/api/scripts/uploads/{upload.id}/review",
            json={"script_contract": VALID_CONTRACT},
        )
        assert r.status_code == 422
        assert "edit_not_allowed" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_concurrent_edit_returns_409(self, client, admin_override):
        script = _make_script(status="draft")
        script.updated_at = datetime(2025, 6, 1, tzinfo=timezone.utc)
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        # Send an old expected_updated_at
        r = await client.patch(
            f"/api/scripts/uploads/{upload.id}/review",
            json={
                "script_contract": VALID_CONTRACT,
                "expected_updated_at": "2025-01-01T00:00:00Z",
            },
        )
        assert r.status_code == 409
        assert "concurrent_modification" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_prohibited_conflict_rejected(self, client, admin_override):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        conflict = {**VALID_CONTRACT}
        conflict["prohibited_responses"] = ["Hey"]  # conflicts with expected_reply
        r = await client.patch(
            f"/api/scripts/uploads/{upload.id}/review",
            json={"script_contract": conflict},
        )
        assert r.status_code == 422
        app.dependency_overrides.pop(get_db, None)

    async def test_audit_event_emitted(self, client, admin_override, caplog):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_edit(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch(
            "app.services.script_registry.update_draft",
            new_callable=AsyncMock,
            return_value=script,
        ):
            with caplog.at_level(logging.INFO, logger="app.api.review"):
                r = await client.patch(
                    f"/api/scripts/uploads/{upload.id}/review",
                    json={"script_contract": VALID_CONTRACT},
                )
                assert r.status_code == 200
                # Find audit record
                audit_records = [
                    rec for rec in caplog.records
                    if rec.getMessage() == "upload_draft_edited"
                ]
                assert len(audit_records) == 1
                rec = audit_records[0]
                assert rec.__dict__["upload_id"] == str(upload.id)
                assert rec.__dict__["script_id"] == str(script.id)
                # No contract content in audit
                assert "Hello" not in str(rec.__dict__)
        app.dependency_overrides.pop(get_db, None)


# ─── Rejection tests ──────────────────────────────────────────────────────────


class TestRejectReview:
    """POST /uploads/{id}/review/reject tests."""

    async def test_valid_rejection(self, client, admin_override):
        upload = _make_upload()
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = upload
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.upload_quarantine.get_quarantine_path") as mock_qp:
            mock_qp.return_value = MagicMock()
            mock_qp.return_value.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )
            r = await client.post(
                f"/api/scripts/uploads/{upload.id}/review/reject",
                json={"reason": "Not suitable for production"},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["review_status"] == "rejected"
            assert data["rejection_reason"] == "Not suitable for production"
            assert data["rejected_by"] == str(admin_override.id)
        app.dependency_overrides.pop(get_db, None)

    async def test_empty_reason_rejected(self, client, admin_override):
        uid = uuid.uuid4()
        r = await client.post(
            f"/api/scripts/uploads/{uid}/review/reject",
            json={"reason": "   "},
        )
        assert r.status_code == 422

    async def test_missing_reason_rejected(self, client, admin_override):
        uid = uuid.uuid4()
        r = await client.post(
            f"/api/scripts/uploads/{uid}/review/reject",
            json={},
        )
        assert r.status_code == 422

    async def test_published_cannot_be_rejected(self, client, admin_override):
        script = _make_script(status="published")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = upload
            elif call_count["n"] == 2:
                result.scalar_one_or_none.return_value = script
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        mock_db.commit = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{upload.id}/review/reject",
            json={"reason": "test"},
        )
        assert r.status_code == 422
        assert "reject_not_allowed" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_already_rejected_idempotent(self, client, admin_override):
        rej_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        rej_by = uuid.uuid4()
        upload = _make_upload(
            status="rejected",
            rejected_at=rej_time,
            rejected_by=rej_by,
            rejection_reason="Previous reason",
        )
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{upload.id}/review/reject",
            json={"reason": "New reason"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["rejection_reason"] == "Previous reason"
        app.dependency_overrides.pop(get_db, None)

    async def test_commit_failure_does_not_delete_source(
        self, client, admin_override
    ):
        """External quarantine cleanup happens only after durable rejection."""
        upload = _make_upload()
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=result)
        mock_db.commit = AsyncMock(side_effect=RuntimeError("db unavailable"))
        mock_db.rollback = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        source = MagicMock()
        source.unlink = MagicMock()
        quarantine = MagicMock()
        quarantine.__truediv__.return_value = source
        with patch(
            "app.services.upload_quarantine.get_quarantine_path",
            return_value=quarantine,
        ):
            response = await client.post(
                f"/api/scripts/uploads/{upload.id}/review/reject",
                json={"reason": "Not suitable"},
            )

        assert response.status_code == 500
        source.unlink.assert_not_called()
        mock_db.rollback.assert_awaited_once()
        app.dependency_overrides.pop(get_db, None)


# ─── Publish tests ────────────────────────────────────────────────────────────


class TestPublishReview:
    """POST /uploads/{id}/review/publish tests."""

    async def test_eligible_draft_publishes(self, client, admin_override):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_review(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_version = MagicMock()
        mock_version.version_number = 1
        mock_version.published_at = datetime(2025, 7, 1, tzinfo=timezone.utc)

        with patch("app.services.script_registry.publish", new_callable=AsyncMock, return_value=mock_version):
            r = await client.post(f"/api/scripts/uploads/{upload.id}/review/publish")
            assert r.status_code == 200
            data = r.json()
            assert data["review_status"] == "published"
            assert data["version_number"] == 1
            assert data["script_status"] == "published"
        app.dependency_overrides.pop(get_db, None)

    async def test_no_linked_script_rejected(self, client, admin_override):
        upload = _make_upload(script_id=None)
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(f"/api/scripts/uploads/{upload.id}/review/publish")
        assert r.status_code == 422
        assert "no_linked_script" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_infected_upload_cannot_publish(self, client, admin_override):
        upload = _make_upload(scan_status="infected", status="failed")
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        # Give it a script_id so it passes the first check
        upload.script_id = uuid.uuid4()
        r = await client.post(f"/api/scripts/uploads/{upload.id}/review/publish")
        assert r.status_code == 422
        assert "publish_not_allowed" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_rejected_upload_cannot_publish(self, client, admin_override):
        upload = _make_upload(status="rejected")
        upload.script_id = uuid.uuid4()
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(f"/api/scripts/uploads/{upload.id}/review/publish")
        assert r.status_code == 422
        app.dependency_overrides.pop(get_db, None)

    async def test_audit_event_emitted_on_publish(self, client, admin_override, caplog):
        script = _make_script(status="draft")
        upload = _make_upload(script_id=script.id)
        from app.database import get_session as get_db
        mock_db = _mock_db_for_review(upload, script)
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_version = MagicMock()
        mock_version.version_number = 1
        mock_version.published_at = datetime(2025, 7, 1, tzinfo=timezone.utc)

        with patch("app.services.script_registry.publish", new_callable=AsyncMock, return_value=mock_version):
            with caplog.at_level(logging.INFO, logger="app.api.review"):
                r = await client.post(f"/api/scripts/uploads/{upload.id}/review/publish")
                assert r.status_code == 200
                audit_records = [
                    rec for rec in caplog.records
                    if rec.getMessage() == "upload_script_published"
                ]
                assert len(audit_records) == 1
                rec = audit_records[0]
                assert rec.__dict__["version_number"] == 1
                # No contract content in audit
                assert "Hello" not in str(rec.__dict__)
        app.dependency_overrides.pop(get_db, None)


# ─── Retry tests ──────────────────────────────────────────────────────────────


class TestRetryReview:
    """POST /uploads/{id}/review/retry tests."""

    async def test_infected_cannot_retry(self, client, admin_override):
        upload = _make_upload(scan_status="infected", status="failed")
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{upload.id}/review/retry",
            json={"target": "scan"},
        )
        assert r.status_code == 422
        assert "infected" in r.json()["detail"]["message"]
        app.dependency_overrides.pop(get_db, None)

    async def test_deleted_cannot_retry(self, client, admin_override):
        upload = _make_upload(status="deleted")
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{upload.id}/review/retry",
            json={"target": "conversion"},
        )
        assert r.status_code == 422
        assert "deleted" in r.json()["detail"]["message"].lower()
        app.dependency_overrides.pop(get_db, None)

    async def test_rejected_cannot_retry(self, client, admin_override):
        upload = _make_upload(status="rejected")
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = upload
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{upload.id}/review/retry",
            json={"target": "conversion"},
        )
        assert r.status_code == 422
        assert "rejected" in r.json()["detail"]["message"].lower()
        app.dependency_overrides.pop(get_db, None)

    async def test_invalid_target_rejected(self, client, admin_override):
        upload = _make_upload()
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = upload
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        mock_db.commit = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{upload.id}/review/retry",
            json={"target": "invalid_step"},
        )
        assert r.status_code == 422
        assert "invalid_target" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)

    async def test_missing_upload_returns_404(self, client, admin_override):
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db

        r = await client.post(
            f"/api/scripts/uploads/{uuid.uuid4()}/review/retry",
            json={"target": "conversion"},
        )
        assert r.status_code == 404
        app.dependency_overrides.pop(get_db, None)

    async def test_scan_retry_missing_source_returns_410(self, client, admin_override):
        upload = _make_upload(scan_status="error", status="failed")
        from app.database import get_session as get_db
        mock_db = AsyncMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = upload
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        mock_db.commit = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.upload_quarantine.get_quarantine_path") as mock_qp:
            mock_path = MagicMock()
            mock_path.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )
            mock_qp.return_value = mock_path
            r = await client.post(
                f"/api/scripts/uploads/{upload.id}/review/retry",
                json={"target": "scan"},
            )
            assert r.status_code == 410
            assert "source_expired" in r.json()["detail"]["error"]
        app.dependency_overrides.pop(get_db, None)
