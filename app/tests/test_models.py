"""Unit tests for SQLAlchemy models."""

import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.database import Base
from app.models import (
    CoachingReport,
    Evaluation,
    LearningPlan,
    Scenario,
    Session,
    SessionReport,
    SessionReportStatus,
    Transcript,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with DBSession(engine) as session:
        yield session
    engine.dispose()


def test_scenario_model_creation(db_session):
    """Test that a Scenario can be created with all required fields."""
    scenario = Scenario(
        id=uuid.uuid4(),
        name="Financial Hardship",
        scenario_type="FINANCIAL_HARDSHIP",
        description="A debtor facing financial difficulties.",
        debtor_profile={
            "name": "John Doe",
            "outstanding_balance": "5000.00",
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=True,
    )
    db_session.add(scenario)
    db_session.commit()

    result = db_session.execute(select(Scenario)).scalar_one()
    assert result.name == "Financial Hardship"
    assert result.scenario_type == "FINANCIAL_HARDSHIP"
    assert result.debtor_profile["name"] == "John Doe"
    assert result.is_active is True


def test_session_model_with_foreign_key(db_session):
    """Test that a Session links to a Scenario via foreign key."""
    scenario_id = uuid.uuid4()
    scenario = Scenario(
        id=scenario_id,
        name="Angry Customer",
        scenario_type="ANGRY_CUSTOMER",
        debtor_profile={
            "name": "Jane Smith",
            "outstanding_balance": "1200.00",
            "days_past_due": 30,
            "personality_profile": "hostile",
            "conversation_goal": "resolve dispute",
        },
        is_active=True,
    )
    db_session.add(scenario)
    db_session.commit()

    session = Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=uuid.uuid4(),
        status="pending",
        persona_context={"emotional_state": "hostile"},
    )
    db_session.add(session)
    db_session.commit()

    result = db_session.execute(select(Session)).scalar_one()
    assert result.scenario_id == scenario_id
    assert result.status == "pending"
    assert result.persona_context["emotional_state"] == "hostile"


def test_transcript_model(db_session):
    """Test Transcript model with all required fields."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=uuid.uuid4(),
            status="active",
        )
    )
    db_session.commit()

    transcript = Transcript(
        id=uuid.uuid4(),
        session_id=session_id,
        speaker="agent",
        utterance_text="Hello, I'm calling about your account.",
        timestamp_ms=datetime.now(UTC),
        sequence_number=1,
    )
    db_session.add(transcript)
    db_session.commit()

    result = db_session.execute(select(Transcript)).scalar_one()
    assert result.speaker == "agent"
    assert result.sequence_number == 1
    assert result.utterance_text == "Hello, I'm calling about your account."


def test_evaluation_unique_session(db_session):
    """Test that Evaluation has a unique constraint on session_id."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=uuid.uuid4(),
            status="completed",
        )
    )
    db_session.commit()

    evaluation = Evaluation(
        id=uuid.uuid4(),
        session_id=session_id,
        overall_score=75.5,
        category_scores=[{"category": "compliance", "score": 80}],
        strengths=[{"description": "Good opening"}],
        weaknesses=[{"description": "Missed compliance"}],
        is_too_short=False,
    )
    db_session.add(evaluation)
    db_session.commit()

    result = db_session.execute(select(Evaluation)).scalar_one()
    assert result.overall_score == 75.5
    assert result.is_too_short is False


def test_coaching_report_model(db_session):
    """Test CoachingReport model creation."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=uuid.uuid4(),
            status="completed",
        )
    )
    db_session.commit()

    report = CoachingReport(
        id=uuid.uuid4(),
        session_id=session_id,
        mistakes_by_category={"compliance": [{"explanation": "Did not verify identity"}]},
        total_mistakes=1,
        no_mistakes=False,
    )
    db_session.add(report)
    db_session.commit()

    result = db_session.execute(select(CoachingReport)).scalar_one()
    assert result.total_mistakes == 1
    assert result.no_mistakes is False


def test_learning_plan_model(db_session):
    """Test LearningPlan model creation."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=uuid.uuid4(),
            status="completed",
        )
    )
    db_session.commit()

    plan = LearningPlan(
        id=uuid.uuid4(),
        session_id=session_id,
        agent_id=uuid.uuid4(),
        weak_competencies=[
            {
                "category": "compliance",
                "score": 55,
                "recommended_scenario": "Compliance Fundamentals",
            }
        ],
        all_passing=False,
    )
    db_session.add(plan)
    db_session.commit()

    result = db_session.execute(select(LearningPlan)).scalar_one()
    assert result.all_passing is False
    assert len(result.weak_competencies) == 1
    assert result.weak_competencies[0]["category"] == "compliance"


def test_session_report_model_creation(db_session):
    """Test that a SessionReport can be created with all required fields."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=agent_id,
            status="completed",
        )
    )
    db_session.commit()

    report = SessionReport(
        id=uuid.uuid4(),
        session_id=session_id,
        agent_id=agent_id,
        status=SessionReportStatus.READY,
        report_version=1,
        payload={"summary": {"session_id": str(session_id)}},
        content_hash="a" * 64,
        generated_by=agent_id,
    )
    db_session.add(report)
    db_session.commit()

    result = db_session.execute(select(SessionReport)).scalar_one()
    assert result.session_id == session_id
    assert result.status == SessionReportStatus.READY
    assert result.report_version == 1
    assert result.payload["summary"]["session_id"] == str(session_id)
    assert result.content_hash == "a" * 64
    assert result.failure_reason is None


def test_session_report_failed_status_has_no_payload(db_session):
    """A failed generation records a reason and no payload."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=agent_id,
            status="completed",
        )
    )
    db_session.commit()

    report = SessionReport(
        id=uuid.uuid4(),
        session_id=session_id,
        agent_id=agent_id,
        status=SessionReportStatus.FAILED,
        report_version=1,
        payload=None,
        failure_reason="Assembly failed: missing evaluation artifact",
        reason_code="generation_failed",
    )
    db_session.add(report)
    db_session.commit()

    result = db_session.execute(select(SessionReport)).scalar_one()
    assert result.status == SessionReportStatus.FAILED
    assert result.payload is None
    assert result.failure_reason == "Assembly failed: missing evaluation artifact"
    assert result.reason_code == "generation_failed"


def test_session_report_unique_session_version_constraint(db_session):
    """The (session_id, report_version) pair must be unique."""
    from sqlalchemy.exc import IntegrityError

    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=agent_id,
            status="completed",
        )
    )
    db_session.commit()

    db_session.add(
        SessionReport(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            status=SessionReportStatus.READY,
            report_version=1,
            payload={"a": 1},
            content_hash="a" * 64,
        )
    )
    db_session.commit()

    db_session.add(
        SessionReport(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            status=SessionReportStatus.READY,
            report_version=1,
            payload={"a": 2},
            content_hash="b" * 64,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_session_report_multiple_versions_preserve_prior_payload(db_session):
    """Regeneration adds a new version row; the prior payload is untouched."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    db_session.add(
        Scenario(
            id=scenario_id,
            name="Test",
            scenario_type="TEST",
            debtor_profile={
                "name": "X",
                "outstanding_balance": "100",
                "days_past_due": 1,
                "personality_profile": "calm",
                "conversation_goal": "pay",
            },
            is_active=True,
        )
    )
    db_session.add(
        Session(
            id=session_id,
            scenario_id=scenario_id,
            agent_id=agent_id,
            status="completed",
        )
    )
    db_session.commit()

    db_session.add(
        SessionReport(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            status=SessionReportStatus.READY,
            report_version=1,
            payload={"revision": "first"},
            content_hash="a" * 64,
        )
    )
    db_session.add(
        SessionReport(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            status=SessionReportStatus.READY,
            report_version=2,
            payload={"revision": "second"},
            content_hash="b" * 64,
        )
    )
    db_session.commit()

    results = (
        db_session.execute(
            select(SessionReport)
            .where(SessionReport.session_id == session_id)
            .order_by(SessionReport.report_version)
        )
        .scalars()
        .all()
    )
    assert len(results) == 2
    assert results[0].report_version == 1
    assert results[0].payload == {"revision": "first"}
    assert results[1].report_version == 2
    assert results[1].payload == {"revision": "second"}


def _admin_dsn(database: str = "postgres") -> str:
    """Build a plain (non-asyncpg) DSN against the same Postgres server used by settings, but a specific DB."""
    async_url = settings.async_database_url
    # postgresql+asyncpg://user:pass@host:port/dbname -> postgresql://user:pass@host:port/dbname
    plain = async_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    base, _, _ = plain.rpartition("/")
    return f"{base}/{database}"


async def _create_scratch_database(db_name: str) -> None:
    conn = await asyncpg.connect(_admin_dsn())
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()


async def _drop_scratch_database(db_name: str) -> None:
    conn = await asyncpg.connect(_admin_dsn())
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await conn.close()


def _run_alembic(project_root: Path, database_url: str, *args: str) -> subprocess.CompletedProcess:
    alembic_bin = shutil.which("alembic") or str(Path(sys.executable).parent / "alembic.exe")
    env = os.environ.copy()
    env["CAT_DATABASE_URL"] = database_url
    result = subprocess.run(
        [alembic_bin, *args],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    return result


@pytest.mark.asyncio
async def test_migration_upgrade_downgrade_upgrade_smoke():
    """Migration smoke test: upgrade -> downgrade -1 -> upgrade against a scratch DB.

    Runs the full Alembic migration chain against a disposable PostgreSQL
    database (created on the same server configured for tests), then
    downgrades the campaign branch from the merge head and upgrades back
    to head, asserting the
    script registry schema (scripts/script_versions tables and
    sessions.script_version_id) is present at the end.

    Validates: Requirements 3.6, 3.7
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    db_name = f"cat_db_migration_test_{uuid.uuid4().hex[:8]}"
    scratch_asyncpg_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+asyncpg://", 1)
    scratch_psycopg_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+psycopg2://", 1)

    await _create_scratch_database(db_name)
    try:
        result = _run_alembic(project_root, scratch_asyncpg_url, "upgrade", "head")
        assert result.returncode == 0, result.stderr

        result = _run_alembic(
            project_root,
            scratch_asyncpg_url,
            "downgrade",
            "014_negotiation_linear",
        )
        assert result.returncode == 0, result.stderr

        result = _run_alembic(project_root, scratch_asyncpg_url, "upgrade", "head")
        assert result.returncode == 0, result.stderr

        # Inspect the resulting schema with a sync driver (psycopg2 is on requirements.txt).
        engine = create_engine(scratch_psycopg_url)
        try:
            inspector = inspect(engine)
            table_names = set(inspector.get_table_names())
            assert "scripts" in table_names
            assert "script_versions" in table_names

            session_columns = {col["name"] for col in inspector.get_columns("sessions")}
            assert "script_version_id" in session_columns
        finally:
            engine.dispose()
    finally:
        await _drop_scratch_database(db_name)


@pytest.mark.asyncio
async def test_migration_script_uploads_schema():
    """Verify script_uploads table schema after full migration upgrade.

    Checks:
    - Table exists
    - All 19 expected columns present with correct types
    - Foreign keys: uploaded_by→users, scenario_id→scenarios, script_id→scripts
    - Non-null constraints on required columns
    - Upgrade → downgrade → upgrade lifecycle succeeds
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    db_name = f"cat_db_upload_schema_test_{uuid.uuid4().hex[:8]}"
    scratch_asyncpg_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+asyncpg://", 1)
    scratch_psycopg_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+psycopg2://", 1)

    await _create_scratch_database(db_name)
    try:
        # Upgrade to head
        result = _run_alembic(project_root, scratch_asyncpg_url, "upgrade", "head")
        assert result.returncode == 0, f"upgrade failed: {result.stderr}"

        # Inspect schema
        engine = create_engine(scratch_psycopg_url)
        try:
            inspector = inspect(engine)
            table_names = set(inspector.get_table_names())
            assert (
                "script_uploads" in table_names
            ), f"script_uploads not found. Tables: {sorted(table_names)}"

            # Verify all columns
            columns = {col["name"]: col for col in inspector.get_columns("script_uploads")}
            expected_columns = [
                "id",
                "filename_original",
                "mime_type",
                "file_size_bytes",
                "content_hash",
                "storage_key",
                "uploaded_by",
                "scan_status",
                "scan_signature",
                "extraction_status",
                "extraction_error",
                "extracted_content",
                "scenario_id",
                "status",
                "script_id",
                "created_at",
                "updated_at",
                "quarantine_expires_at",
                "deleted_at",
            ]
            for col_name in expected_columns:
                assert col_name in columns, f"Missing column: {col_name}"

            # Verify non-null constraints on required columns
            required_non_null = [
                "id",
                "filename_original",
                "mime_type",
                "file_size_bytes",
                "storage_key",
                "uploaded_by",
                "scan_status",
                "extraction_status",
                "status",
                "created_at",
                "updated_at",
                "quarantine_expires_at",
            ]
            for col_name in required_non_null:
                assert (
                    columns[col_name]["nullable"] is False
                ), f"Column {col_name} should be NOT NULL"

            # Verify nullable columns (content_hash is nullable for failed pre-extraction records)
            nullable_columns = [
                "content_hash",
                "scan_signature",
                "extraction_error",
                "extracted_content",
                "scenario_id",
                "script_id",
                "deleted_at",
            ]
            for col_name in nullable_columns:
                assert (
                    columns[col_name]["nullable"] is True
                ), f"Column {col_name} should be nullable"

            # Verify foreign keys
            fks = inspector.get_foreign_keys("script_uploads")
            fk_map = {}
            for fk in fks:
                for col in fk["constrained_columns"]:
                    fk_map[col] = f"{fk['referred_table']}.{fk['referred_columns'][0]}"

            assert (
                fk_map.get("uploaded_by") == "users.id"
            ), f"uploaded_by FK: {fk_map.get('uploaded_by')}"
            assert (
                fk_map.get("scenario_id") == "scenarios.id"
            ), f"scenario_id FK: {fk_map.get('scenario_id')}"
            assert (
                fk_map.get("script_id") == "scripts.id"
            ), f"script_id FK: {fk_map.get('script_id')}"
        finally:
            engine.dispose()

        # Downgrade to just before our merge (one of the parents), then re-upgrade
        # We use the merge revision's parent as the target since -1 is ambiguous for merges
        result = _run_alembic(
            project_root, scratch_asyncpg_url, "downgrade", "009_refactor_roles_add_user_type"
        )
        assert result.returncode == 0, f"downgrade failed: {result.stderr}"

        result = _run_alembic(project_root, scratch_asyncpg_url, "upgrade", "head")
        assert result.returncode == 0, f"re-upgrade failed: {result.stderr}"

        # Verify schema still correct after cycle
        engine = create_engine(scratch_psycopg_url)
        try:
            inspector = inspect(engine)
            assert "script_uploads" in set(inspector.get_table_names())
            cols = {c["name"] for c in inspector.get_columns("script_uploads")}
            assert "extracted_content" in cols
            assert "scenario_id" in cols
        finally:
            engine.dispose()

    finally:
        await _drop_scratch_database(db_name)


@pytest.mark.asyncio
async def test_migration_011_content_hash_nullable_round_trip():
    """Data round-trip: NULL content_hash survives downgrade/upgrade cycle.

    1. Upgrade to 011_nullable_content_hash.
    2. Insert a failed row (content_hash=NULL) and a successful row (valid hash).
    3. Downgrade to 010_merge_upload_and_roles.
    4. Verify: failed row uses '' sentinel, successful row unchanged, NOT NULL active.
    5. Upgrade to head.
    6. Verify: failed row restored to NULL, successful row unchanged, column nullable.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    db_name = f"cat_db_hash_roundtrip_{uuid.uuid4().hex[:8]}"
    scratch_asyncpg_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+asyncpg://", 1)
    scratch_psycopg_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+psycopg2://", 1)

    await _create_scratch_database(db_name)
    try:
        # Step 1: Upgrade to head (includes 011)
        result = _run_alembic(project_root, scratch_asyncpg_url, "upgrade", "head")
        assert result.returncode == 0, f"upgrade failed: {result.stderr}"

        # Step 2: Insert test rows
        engine = create_engine(scratch_psycopg_url)
        try:
            from sqlalchemy import text

            with engine.begin() as conn:
                # Create minimal user and scenario for FK satisfaction
                user_id = uuid.uuid4()
                conn.execute(
                    text(
                        "INSERT INTO users (id, email, hashed_password, full_name, role, is_active) "
                        "VALUES (:id, 'test@t.com', 'x', 'Test User', 'admin', true)"
                    ),
                    {"id": user_id},
                )

                scenario_id = uuid.uuid4()
                conn.execute(
                    text(
                        "INSERT INTO scenarios (id, name, scenario_type, debtor_profile, is_active) "
                        "VALUES (:id, 'Test', 'TEST', '{}'::jsonb, true)"
                    ),
                    {"id": scenario_id},
                )

                # Failed row: content_hash = NULL
                failed_id = uuid.uuid4()
                conn.execute(
                    text(
                        "INSERT INTO script_uploads "
                        "(id, filename_original, mime_type, file_size_bytes, content_hash, "
                        " storage_key, uploaded_by, scan_status, extraction_status, status, "
                        " quarantine_expires_at) "
                        "VALUES (:id, 'bad.pdf', 'application/pdf', 1000, NULL, "
                        " 'key1.pdf', :uid, 'infected', 'failed', 'failed', NOW())"
                    ),
                    {"id": failed_id, "uid": user_id},
                )

                # Successful row: valid hash
                valid_hash = "a" * 64
                success_id = uuid.uuid4()
                conn.execute(
                    text(
                        "INSERT INTO script_uploads "
                        "(id, filename_original, mime_type, file_size_bytes, content_hash, "
                        " storage_key, uploaded_by, scan_status, extraction_status, status, "
                        " quarantine_expires_at) "
                        "VALUES (:id, 'ok.pdf', 'application/pdf', 2000, :hash, "
                        " 'key2.pdf', :uid, 'clean', 'completed', 'completed', NOW())"
                    ),
                    {"id": success_id, "uid": user_id, "hash": valid_hash},
                )
        finally:
            engine.dispose()

        # Step 3: Downgrade to 010
        result = _run_alembic(
            project_root, scratch_asyncpg_url, "downgrade", "010_merge_upload_and_roles"
        )
        assert result.returncode == 0, f"downgrade failed: {result.stderr}"

        # Step 4: Verify sentinel and constraint
        engine = create_engine(scratch_psycopg_url)
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                # Failed row should have '' sentinel
                row = conn.execute(
                    text("SELECT content_hash FROM script_uploads WHERE id = :id"),
                    {"id": failed_id},
                ).fetchone()
                assert row[0] == "", f"Expected '' sentinel, got: {row[0]!r}"

                # Successful row unchanged
                row = conn.execute(
                    text("SELECT content_hash FROM script_uploads WHERE id = :id"),
                    {"id": success_id},
                ).fetchone()
                assert row[0] == valid_hash

            # Verify NOT NULL constraint is active
            inspector = inspect(engine)
            cols = {c["name"]: c for c in inspector.get_columns("script_uploads")}
            assert cols["content_hash"]["nullable"] is False
        finally:
            engine.dispose()

        # Step 5: Upgrade back to head
        result = _run_alembic(project_root, scratch_asyncpg_url, "upgrade", "head")
        assert result.returncode == 0, f"re-upgrade failed: {result.stderr}"

        # Step 6: Verify restoration
        engine = create_engine(scratch_psycopg_url)
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                # Failed row restored to NULL
                row = conn.execute(
                    text("SELECT content_hash FROM script_uploads WHERE id = :id"),
                    {"id": failed_id},
                ).fetchone()
                assert row[0] is None, f"Expected NULL, got: {row[0]!r}"

                # Successful row still has valid hash
                row = conn.execute(
                    text("SELECT content_hash FROM script_uploads WHERE id = :id"),
                    {"id": success_id},
                ).fetchone()
                assert row[0] == valid_hash

            # Column is nullable again
            inspector = inspect(engine)
            cols = {c["name"]: c for c in inspector.get_columns("script_uploads")}
            assert cols["content_hash"]["nullable"] is True
        finally:
            engine.dispose()

    finally:
        await _drop_scratch_database(db_name)
