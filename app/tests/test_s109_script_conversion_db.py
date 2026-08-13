"""S1-09: Real PostgreSQL database integration tests for script conversion.

Uses the production conversion service (convert_upload_to_script_draft) with
real AsyncSession objects, real flush/commit/rollback, real unique constraints,
and real row locking. No mocks on database operations.

Requirements:
    - PostgreSQL running
    - Set CAT_TEST_DATABASE_URL to a test-only database, e.g.:
      CAT_TEST_DATABASE_URL=postgresql+asyncpg://local_test_user:local_test_password@localhost:5433/cat_test
    - The database name MUST contain 'test' (e.g. cat_test, testing_db)
    - Start with: docker compose up -d db
    - Create the test DB: createdb -h localhost -p 5433 -U postgres cat_test
    - Run with:
      $env:CAT_TEST_DATABASE_URL="postgresql+asyncpg://local_test_user:local_test_password@localhost:5433/cat_test"
      python -m pytest app/tests/test_s109_script_conversion_db.py -q --tb=short -m db_integration
"""

import asyncio
import json
import os
import re
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Script, ScriptUpload, UploadStatus
from app.models.script import ScriptStatus
from app.models.user import User, UserRole
from app.services.conversion_service import (
    ConversionConflict,
    ConversionInternalError,
    ConversionRejection,
    ConversionSuccess,
    convert_upload_to_script_draft,
    is_scenario_script_unique_violation,
)


# --- Test database URL validation ---

_TEST_DB_URL = os.environ.get("CAT_TEST_DATABASE_URL", "")


def _test_db_url_valid() -> bool:
    """Check if CAT_TEST_DATABASE_URL is set and points to a test database."""
    if not _TEST_DB_URL:
        return False
    if "postgresql" not in _TEST_DB_URL:
        return False
    # Require database name to contain 'test'
    match = re.search(r"/([^/?]+)(\?|$)", _TEST_DB_URL)
    if not match:
        return False
    db_name = match.group(1)
    return "test" in db_name.lower()


_SKIP_REASON = (
    "CAT_TEST_DATABASE_URL is required for PostgreSQL integration tests. "
    "Set it to a test-only database (name must contain 'test'), e.g.: "
    "CAT_TEST_DATABASE_URL=postgresql+asyncpg://local_test_user:local_test_password@localhost:5433/cat_test"
)

pytestmark = [
    pytest.mark.db_integration,
    pytest.mark.skipif(not _test_db_url_valid(), reason=_SKIP_REASON),
]


# --- Valid contract data ---

VALID_CONTRACT = {
    "debtor_persona": {
        "name": "DB Test",
        "communication_style": "Direct",
        "background": "Test scenario",
    },
    "financial_situation": {
        "outstanding_balance": "1500.00",
        "days_past_due": 15,
        "reason_for_delinquency": "Job loss",
    },
    "opening_response": "Hello, I received your letter.",
    "expected_replies": [{"agent_statement": "Can you confirm?", "debtor_reply": "Yes confirmed"}],
    "trigger_phrases": [{"phrase": "payment plan", "behavior": "express interest"}],
    "emotional_state_rules": [{"trigger": "harsh tone", "state_change": "become guarded"}],
    "payment_conditions": [{"condition": "full amount", "term": "monthly", "accepted": False}],
    "escalation_conditions": [{"condition": "threats", "behavior": "end call", "ends_call": True}],
    "prohibited_responses": ["I refuse everything"],
    "conversation_goal": {
        "target_outcome": "payment arrangement",
        "completion_condition": "verbal agreement",
    },
}


# --- Fixtures ---


@pytest.fixture
async def engine():
    """Create engine and ensure tables exist in test database."""
    eng = create_async_engine(_TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session_factory(engine):
    """Session factory for the test database."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db(session_factory):
    """Session for fixture setup. Commits then cleans up after test."""
    async with session_factory() as session:
        yield session


@pytest.fixture
async def admin_user(db):
    """Persist a real admin user and clean up after test."""
    user = User(
        id=uuid.uuid4(),
        email=f"admin-{uuid.uuid4().hex[:8]}@test.com",
        full_name="Test Admin",
        role=UserRole.ADMIN.value,
        is_active=True,
        hashed_password="$2b$12$fakehashfortest",
        auth_provider="local",
    )
    db.add(user)
    await db.commit()
    yield user
    # Cleanup
    await db.execute(delete(User).where(User.id == user.id))
    await db.commit()


@pytest.fixture
async def scenario(db, admin_user):
    """Persist a scenario; clean up after test."""
    s = Scenario(
        id=uuid.uuid4(),
        name=f"Test Scenario {uuid.uuid4().hex[:6]}",
        scenario_type="debt_collection",
        description="Integration test scenario",
        debtor_profile={"name": "Test Debtor"},
    )
    db.add(s)
    await db.commit()
    yield s
    # Cleanup: clear upload FK references first, then scripts, then uploads, then scenario
    await db.execute(
        text("UPDATE script_uploads SET script_id = NULL WHERE scenario_id = :sid"),
        {"sid": s.id},
    )
    await db.execute(delete(Script).where(Script.scenario_id == s.id))
    await db.execute(delete(ScriptUpload).where(ScriptUpload.scenario_id == s.id))
    await db.execute(delete(Scenario).where(Scenario.id == s.id))
    await db.commit()


def _make_clean_upload(admin_id, scenario_id, content=None):
    """Build a ScriptUpload instance (not persisted)."""
    return ScriptUpload(
        id=uuid.uuid4(),
        filename_original="test_contract.json",
        mime_type="application/json",
        file_size_bytes=len(json.dumps(VALID_CONTRACT)),
        content_hash=f"hash_{uuid.uuid4().hex[:8]}",
        storage_key=f"{uuid.uuid4()}.json",
        uploaded_by=admin_id,
        scan_status="clean",
        extraction_status="completed",
        extracted_content=content or json.dumps(VALID_CONTRACT),
        scenario_id=scenario_id,
        status=UploadStatus.COMPLETED.value,
        quarantine_expires_at=datetime.now(UTC) + timedelta(hours=24),
    )


# ============================================================
# REAL DATABASE TESTS — using production service
# ============================================================


class TestAtomicSuccess:
    """Conversion creates Script and links upload atomically."""

    async def test_conversion_creates_script_and_links(
        self, session_factory, admin_user, scenario, db
    ):
        upload = _make_clean_upload(admin_user.id, scenario.id)
        db.add(upload)
        await db.commit()

        # Execute conversion via the production service
        async with session_factory() as session:
            result = await convert_upload_to_script_draft(session, upload.id, admin_user.id)

        assert isinstance(result, ConversionSuccess)

        # Verify from a separate session
        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 1
            assert scripts[0].draft_content["opening_response"] == "Hello, I received your letter."

            u = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == upload.id))
            ).scalar_one()
            assert u.script_id == scripts[0].id
            assert result.script_id == scripts[0].id


class TestConversionFailure:
    """Invalid content: no Script, script_id stays null."""

    async def test_partial_contract_no_script(self, session_factory, admin_user, scenario, db):
        partial = json.dumps(
            {"debtor_persona": {"name": "X", "communication_style": "Y", "background": "Z"}}
        )
        upload = _make_clean_upload(admin_user.id, scenario.id, content=partial)
        db.add(upload)
        await db.commit()

        async with session_factory() as session:
            result = await convert_upload_to_script_draft(session, upload.id, admin_user.id)

        assert isinstance(result, ConversionRejection)
        assert result.error == "conversion_failed"

        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 0
            u = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == upload.id))
            ).scalar_one()
            assert u.script_id is None


class TestLimitFailure:
    """Content exceeding limits: no Script."""

    async def test_too_many_triggers_rejected(self, session_factory, admin_user, scenario, db):
        data = {**VALID_CONTRACT}
        data["trigger_phrases"] = [{"phrase": f"p{i}", "behavior": f"b{i}"} for i in range(51)]
        upload = _make_clean_upload(admin_user.id, scenario.id, content=json.dumps(data))
        db.add(upload)
        await db.commit()

        async with session_factory() as session:
            result = await convert_upload_to_script_draft(session, upload.id, admin_user.id)

        assert isinstance(result, ConversionRejection)
        assert "publication requirements" in result.message

        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 0


class TestExistingScenarioScript:
    """Scenario already has a Script → conflict."""

    async def test_existing_script_conflict(self, session_factory, admin_user, scenario, db):
        # Create existing script
        existing = Script(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            name="Existing",
            status=ScriptStatus.DRAFT.value,
            format="json",
            draft_content=VALID_CONTRACT,
            created_by=admin_user.id,
        )
        db.add(existing)
        upload = _make_clean_upload(admin_user.id, scenario.id)
        db.add(upload)
        await db.commit()

        async with session_factory() as session:
            result = await convert_upload_to_script_draft(session, upload.id, admin_user.id)

        assert isinstance(result, ConversionConflict)
        assert result.error == "scenario_has_script"

        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 1
            assert scripts[0].id == existing.id  # unchanged
            u = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == upload.id))
            ).scalar_one()
            assert u.script_id is None


class TestRepeatedConversion:
    """Second attempt returns already_converted."""

    async def test_second_attempt_conflict(self, session_factory, admin_user, scenario, db):
        upload = _make_clean_upload(admin_user.id, scenario.id)
        db.add(upload)
        await db.commit()

        async with session_factory() as s1:
            r1 = await convert_upload_to_script_draft(s1, upload.id, admin_user.id)
        assert isinstance(r1, ConversionSuccess)

        async with session_factory() as s2:
            r2 = await convert_upload_to_script_draft(s2, upload.id, admin_user.id)
        assert isinstance(r2, ConversionConflict)
        assert r2.error == "already_converted"

        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 1


class TestSameUploadConcurrency:
    """Two simultaneous conversions of same upload — one wins."""

    async def test_concurrent_same_upload(self, session_factory, admin_user, scenario, db):
        upload = _make_clean_upload(admin_user.id, scenario.id)
        db.add(upload)
        await db.commit()

        results = []

        async def attempt():
            async with session_factory() as session:
                r = await convert_upload_to_script_draft(session, upload.id, admin_user.id)
                results.append(r)

        await asyncio.gather(attempt(), attempt())

        successes = [r for r in results if isinstance(r, ConversionSuccess)]
        conflicts = [r for r in results if isinstance(r, ConversionConflict)]
        assert len(successes) == 1, f"Expected 1 success: {results}"
        assert len(conflicts) == 1, f"Expected 1 conflict: {results}"
        assert conflicts[0].error == "already_converted"

        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 1
            u = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == upload.id))
            ).scalar_one()
            assert u.script_id == scripts[0].id


class TestSameScenarioConcurrency:
    """Two uploads for same scenario concurrently — one wins via constraint."""

    async def test_concurrent_same_scenario(self, session_factory, admin_user, scenario, db):
        u1 = _make_clean_upload(admin_user.id, scenario.id)
        u2 = _make_clean_upload(admin_user.id, scenario.id)
        db.add(u1)
        db.add(u2)
        await db.commit()

        results = []

        async def attempt(upload_id):
            async with session_factory() as session:
                try:
                    r = await convert_upload_to_script_draft(session, upload_id, admin_user.id)
                    results.append((upload_id, r))
                except ConversionInternalError as e:
                    results.append((upload_id, e))

        await asyncio.gather(attempt(u1.id), attempt(u2.id))

        successes = [(uid, r) for uid, r in results if isinstance(r, ConversionSuccess)]
        conflicts = [(uid, r) for uid, r in results if isinstance(r, ConversionConflict)]
        errors = [(uid, r) for uid, r in results if isinstance(r, ConversionInternalError)]

        assert len(successes) == 1, f"Expected 1 success: {results}"
        assert len(conflicts) == 1, f"Expected 1 conflict: {results}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert conflicts[0][1].error == "scenario_has_script"

        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 1

            winner_id = successes[0][0]
            loser_id = conflicts[0][0]
            winner = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == winner_id))
            ).scalar_one()
            loser = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == loser_id))
            ).scalar_one()
            assert winner.script_id == scripts[0].id
            assert loser.script_id is None


class TestRollbackVisibility:
    """Rollback verified from separate session."""

    async def test_failed_commit_leaves_nothing(self, session_factory, admin_user, scenario, db):
        """Simulate failure by rolling back after flush — verify nothing persists."""
        from app.services.script_converter import convert_extracted_to_contract
        from app.services.script_registry import create_draft_in_transaction

        upload = _make_clean_upload(admin_user.id, scenario.id)
        db.add(upload)
        await db.commit()

        async with session_factory() as session:
            stmt = select(ScriptUpload).where(ScriptUpload.id == upload.id).with_for_update()
            result = await session.execute(stmt)
            u = result.scalar_one()
            contract_data = convert_extracted_to_contract(u.extracted_content)
            raw = json.dumps(contract_data)
            script = await create_draft_in_transaction(
                session,
                admin_id=admin_user.id,
                name="will rollback",
                scenario_id=scenario.id,
                format="json",
                raw_definition=raw,
            )
            u.script_id = script.id
            # Simulate failure: rollback instead of commit
            await session.rollback()

        # Verify from separate session: nothing persisted
        async with session_factory() as verify:
            scripts = (
                (await verify.execute(select(Script).where(Script.scenario_id == scenario.id)))
                .scalars()
                .all()
            )
            assert len(scripts) == 0
            u_check = (
                await verify.execute(select(ScriptUpload).where(ScriptUpload.id == upload.id))
            ).scalar_one()
            assert u_check.script_id is None


class TestCleanupVerification:
    """Verify test cleanup — no leftover rows from previous tests."""

    async def test_no_leftover_test_scripts(self, session_factory):
        """Ensure cleanup works: no scripts with 'DB Test' persona in test DB."""
        async with session_factory() as verify:
            # Check there are no orphaned scripts from failed test runs
            # (This validates the fixture cleanup strategy)
            result = await verify.execute(
                text("SELECT COUNT(*) FROM scripts WHERE name LIKE '%will rollback%'")
            )
            count = result.scalar()
            assert count == 0


# ============================================================
# CLASSIFIER TESTS — exact constraint matching
# ============================================================


class TestIntegrityErrorClassifier:
    """Test is_scenario_script_unique_violation with exact constraint name."""

    def test_exact_scripts_scenario_id_key_true(self):
        """PostgreSQL unique violation on scripts_scenario_id_key → True."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = "scripts_scenario_id_key"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is True

    def test_other_table_scenario_id_constraint_false(self):
        """Another table's scenario_id unique constraint → False."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = "campaign_scenarios_scenario_id_key"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_different_constraint_false(self):
        """Unique violation on a completely different constraint → False."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = "uq_users_email"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_foreign_key_violation_false(self):
        """Foreign-key violation (23503) → False."""

        class FakeOrig:
            sqlstate = "23503"
            constraint_name = "scripts_scenario_id_fkey"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_not_null_violation_false(self):
        """Not-null violation (23502) → False."""

        class FakeOrig:
            sqlstate = "23502"
            constraint_name = None

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_no_orig_false(self):
        """No original exception → False."""
        exc = IntegrityError("some error", {}, None)
        assert is_scenario_script_unique_violation(exc) is False

    def test_unknown_driver_false(self):
        """Unknown driver structure → False."""

        class FakeOrig:
            pass

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_message_fallback_exact_match(self):
        """Message contains exact constraint name → True."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None

            def __str__(self):
                return 'duplicate key value violates unique constraint "scripts_scenario_id_key"'

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is True

    def test_message_partial_substring_false(self):
        """Message contains 'scenario_id' but not the exact constraint → False."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None

            def __str__(self):
                return 'duplicate key value violates unique constraint "other_scenario_id_key"'

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    # --- Deceptive constraint-name regression tests ---

    def test_deceptive_other_scripts_prefix_false(self):
        """'other_scripts_scenario_id_key' must NOT match (prefix deception)."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = "other_scripts_scenario_id_key"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_deceptive_archived_scripts_prefix_false(self):
        """'archived_scripts_scenario_id_key' must NOT match."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = "archived_scripts_scenario_id_key"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_deceptive_backup_suffix_false(self):
        """'scripts_scenario_id_key_backup' must NOT match (suffix deception)."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = "scripts_scenario_id_key_backup"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_deceptive_message_contains_as_longer_identifier_false(self):
        """Message containing the constraint as part of a longer name must NOT match."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None

            def __str__(self):
                return (
                    'duplicate key value violates unique constraint "other_scripts_scenario_id_key"'
                )

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_deceptive_message_backup_suffix_false(self):
        """Message containing backup-suffixed constraint must NOT match."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None

            def __str__(self):
                return 'duplicate key value violates unique constraint "scripts_scenario_id_key_backup"'

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_unquoted_ambiguous_message_false(self):
        """Unquoted ambiguous message without parseable constraint → False."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None

            def __str__(self):
                return "unique violation on scripts_scenario_id_key related column"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_sqlstate_not_23505_with_matching_constraint_false(self):
        """Even if constraint name matches, wrong SQLSTATE → False."""

        class FakeOrig:
            sqlstate = "23503"  # FK violation, not unique
            constraint_name = "scripts_scenario_id_key"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_psycopg_diag_exact_match_true(self):
        """psycopg diag field with exact constraint name → True."""

        class FakeDiag:
            constraint_name = "scripts_scenario_id_key"

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None  # asyncpg field absent
            diag = FakeDiag()

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is True

    def test_psycopg_diag_different_constraint_false(self):
        """psycopg diag field with different constraint → False."""

        class FakeDiag:
            constraint_name = "other_scripts_scenario_id_key"

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None
            diag = FakeDiag()

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False

    def test_no_constraint_metadata_no_quotes_in_message_false(self):
        """No structured metadata, no quoted identifier in message → False."""

        class FakeOrig:
            sqlstate = "23505"
            constraint_name = None

            def __str__(self):
                return "some unique constraint violation occurred"

        exc = IntegrityError("", {}, FakeOrig())
        assert is_scenario_script_unique_violation(exc) is False
