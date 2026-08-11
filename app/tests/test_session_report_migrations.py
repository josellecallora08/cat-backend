"""Migration coverage for the additive session-report hardening revision."""

import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.models import (
    Campaign,
    CoachingReport,
    Evaluation,
    LearningPlan,
    NegotiationStandard,
    NegotiationStandardVersion,
    Scenario,
    Session,
    Transcript,
    User,
)


def _admin_dsn(database: str = "postgres") -> str:
    plain = settings.database_url.replace("+asyncpg", "")
    base, _, _ = plain.rpartition("/")
    return f"{base}/{database}"


async def _create_scratch_database(db_name: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(_admin_dsn())
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()


async def _drop_scratch_database(db_name: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(_admin_dsn())
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await conn.close()


def _run_alembic(project_root: Path, database_url: str, *args: str) -> subprocess.CompletedProcess:
    alembic_bin = shutil.which("alembic") or str(Path(sys.executable).parent / "alembic.exe")
    env = os.environ.copy()
    env["CAT_DATABASE_URL"] = database_url
    return subprocess.run(
        [alembic_bin, *args],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )


def _seed_representative_rows(engine):
    """Create existing session/artifact data before applying revision 017."""
    agent_id = uuid.uuid4()
    scenario_id = uuid.uuid4()
    campaign_id = uuid.uuid4()
    standard_id = uuid.uuid4()
    standard_version_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    with DBSession(engine) as db:
        db.add(
            User(
                id=agent_id,
                email=f"migration-{agent_id}@example.test",
                hashed_password="not-used",
                full_name="Migration Agent",
                role="user",
                user_type="agent",
                is_active=True,
            )
        )
        db.add(
            Scenario(
                id=scenario_id,
                name=f"Migration Scenario {scenario_id}",
                scenario_type="TEST",
                debtor_profile={"name": "Existing Debtor"},
                is_active=True,
            )
        )
        db.add(
            Campaign(
                id=campaign_id,
                name=f"Migration Campaign {campaign_id}",
                status="active",
            )
        )
        db.add(
            NegotiationStandard(
                id=standard_id,
                campaign_id=campaign_id,
                name="Existing Standard",
                status="published",
                overall_passing_score=70,
                created_by=agent_id,
                updated_by=agent_id,
            )
        )
        db.add(
            NegotiationStandardVersion(
                id=standard_version_id,
                standard_id=standard_id,
                version_number=1,
                schema_version=1,
                snapshot={"criteria": []},
                content_hash="b" * 64,
                created_by=agent_id,
                published_by=agent_id,
            )
        )
        db.add(
            Session(
                id=session_id,
                scenario_id=scenario_id,
                agent_id=agent_id,
                campaign_id=campaign_id,
                negotiation_standard_version_id=standard_version_id,
                status="completed",
                created_at=now,
                ended_at=now,
            )
        )
        db.add(
            Transcript(
                id=uuid.uuid4(),
                session_id=session_id,
                speaker="agent",
                utterance_text="Hello",
                timestamp_ms=now,
                sequence_number=1,
            )
        )
        db.add(
            Evaluation(
                id=uuid.uuid4(),
                session_id=session_id,
                overall_score=80,
                category_scores=[{"category": "opening", "score": 80}],
                strengths=[{"description": "Clear opening"}],
                weaknesses=[],
                negotiation_standard_version_id=standard_version_id,
                is_too_short=False,
            )
        )
        db.add(
            CoachingReport(
                id=uuid.uuid4(),
                session_id=session_id,
                mistakes_by_category={},
                total_mistakes=0,
                no_mistakes=True,
            )
        )
        db.add(
            LearningPlan(
                id=uuid.uuid4(),
                session_id=session_id,
                agent_id=agent_id,
                weak_competencies=[],
                all_passing=True,
            )
        )
        db.commit()

    return {
        "agent_id": agent_id,
        "campaign_id": campaign_id,
        "session_id": session_id,
        "standard_version_id": standard_version_id,
    }


def _seed_baseline_reports(engine, ids):
    """Insert rows in the exact 016 shape, before reason_code exists."""
    insert = text(
        "INSERT INTO session_reports "
        "(id, session_id, agent_id, campaign_id, negotiation_standard_version_id, "
        " status, report_version, payload, content_hash, generated_by, failure_reason) "
        "VALUES (:id, :session_id, :agent_id, :campaign_id, :standard_version_id, "
        " :status, :version, CAST(:payload AS jsonb), :content_hash, :generated_by, :failure_reason)"
    )
    with engine.begin() as conn:
        conn.execute(
            insert,
            {
                "id": uuid.uuid4(),
                **ids,
                "status": "ready",
                "version": 1,
                "payload": '{"summary":{"preserved":true}}',
                "content_hash": "a" * 64,
                "generated_by": ids["agent_id"],
                "failure_reason": None,
            },
        )
        conn.execute(
            insert,
            {
                "id": uuid.uuid4(),
                **ids,
                "status": "pending",
                "version": 2,
                "payload": None,
                "content_hash": None,
                "generated_by": ids["agent_id"],
                "failure_reason": None,
            },
        )
        conn.execute(
            insert,
            {
                "id": uuid.uuid4(),
                **ids,
                "status": "failed",
                "version": 3,
                "payload": None,
                "content_hash": None,
                "generated_by": ids["agent_id"],
                "failure_reason": "legacy safe failure",
            },
        )


@pytest.mark.asyncio
async def test_session_report_hardening_upgrade_downgrade_preserves_data():
    """017 hardens 016 rows and its one-step downgrade preserves report data."""
    project_root = Path(__file__).resolve().parent.parent.parent
    db_name = f"cat_db_report_migration_{uuid.uuid4().hex[:8]}"
    async_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+asyncpg://", 1)
    sync_url = _admin_dsn(db_name).replace("postgresql://", "postgresql+psycopg2://", 1)

    await _create_scratch_database(db_name)
    try:
        result = _run_alembic(project_root, async_url, "upgrade", "016_add_session_reports")
        assert result.returncode == 0, result.stderr

        engine = create_engine(sync_url)
        ids = _seed_representative_rows(engine)
        _seed_baseline_reports(engine, ids)

        result = _run_alembic(project_root, async_url, "upgrade", "head")
        assert result.returncode == 0, result.stderr

        inspector = inspect(engine)
        columns = {column["name"] for column in inspector.get_columns("session_reports")}
        assert "reason_code" in columns
        assert "uq_session_reports_session_version" in {
            constraint["name"] for constraint in inspector.get_unique_constraints("session_reports")
        }
        assert {index["name"] for index in inspector.get_indexes("session_reports")} >= {
            "ix_session_reports_session_id_version_desc",
            "ix_session_reports_campaign_id",
        }
        assert {check["name"] for check in inspector.get_check_constraints("session_reports")} >= {
            "ck_session_reports_status",
            "ck_session_reports_reason_code",
            "ck_session_reports_positive_version",
            "ck_session_reports_hash_length",
            "ck_session_reports_status_payload",
        }
        foreign_keys = {
            foreign_key["name"]: foreign_key["options"].get("ondelete")
            for foreign_key in inspector.get_foreign_keys("session_reports")
        }
        assert foreign_keys == {
            "fk_session_reports_session_id": "CASCADE",
            "fk_session_reports_campaign_id": "SET NULL",
            "fk_session_reports_standard_version_id": "RESTRICT",
        }

        with engine.connect() as conn:
            reasons = conn.execute(
                text(
                    "SELECT status, reason_code, payload, content_hash "
                    "FROM session_reports ORDER BY report_version"
                )
            ).all()
            assert [(row.status, row.reason_code) for row in reasons] == [
                ("ready", None),
                ("pending", "generation_pending"),
                ("failed", "generation_failed"),
            ]
            assert conn.execute(text("SELECT count(*) FROM sessions")).scalar_one() >= 1
            assert conn.execute(text("SELECT count(*) FROM campaigns")).scalar_one() >= 1
            assert conn.execute(text("SELECT count(*) FROM negotiation_standard_versions")).scalar_one() >= 1
            assert conn.execute(text("SELECT count(*) FROM transcripts")).scalar_one() >= 1
            assert conn.execute(text("SELECT count(*) FROM evaluations")).scalar_one() >= 1
            assert conn.execute(text("SELECT count(*) FROM coaching_reports")).scalar_one() >= 1
            assert conn.execute(text("SELECT count(*) FROM learning_plans")).scalar_one() >= 1

        # Every new invariant is exercised against the same valid FK row.
        valid = {
            "id": uuid.uuid4(),
            "session_id": ids["session_id"],
            "agent_id": ids["agent_id"],
            "campaign_id": ids["campaign_id"],
            "standard_version_id": ids["standard_version_id"],
        }
        invalid_cases = [
            {**valid, "status": "unknown", "version": 4, "payload": None, "hash": None, "reason": None},
            {**valid, "status": "ready", "version": 0, "payload": "{}", "hash": "a" * 64, "reason": None},
            {**valid, "status": "ready", "version": 4, "payload": "{}", "hash": "a" * 63, "reason": None},
            {**valid, "status": "ready", "version": 4, "payload": None, "hash": "a" * 64, "reason": None},
            {**valid, "status": "pending", "version": 4, "payload": "{}", "hash": None, "reason": "generation_pending"},
            {**valid, "status": "failed", "version": 4, "payload": None, "hash": None, "reason": "no_evidence"},
            {**valid, "status": "ready", "version": 4, "payload": "{}", "hash": "a" * 64, "reason": "not-a-code"},
        ]
        insert_invalid = text(
            "INSERT INTO session_reports "
            "(id, session_id, agent_id, campaign_id, negotiation_standard_version_id, status, "
            "report_version, payload, content_hash, reason_code) VALUES "
            "(:id, :session_id, :agent_id, :campaign_id, :standard_version_id, :status, "
            ":version, CAST(:payload AS jsonb), :hash, :reason)"
        )
        for case in invalid_cases:
            with pytest.raises(IntegrityError):
                with engine.begin() as conn:
                    conn.execute(insert_invalid, case)

        result = _run_alembic(project_root, async_url, "downgrade", "-1")
        assert result.returncode == 0, result.stderr
        inspector = inspect(engine)
        assert "reason_code" not in {column["name"] for column in inspector.get_columns("session_reports")}
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM session_reports")).scalar_one() == 3
            assert conn.execute(
                text("SELECT payload FROM session_reports WHERE report_version = 1")
            ).scalar_one() == {"summary": {"preserved": True}}

        result = _run_alembic(project_root, async_url, "upgrade", "head")
        assert result.returncode == 0, result.stderr
        with engine.connect() as conn:
            assert conn.execute(
                text(
                    "SELECT reason_code FROM session_reports "
                    "WHERE status = 'failed' AND report_version = 3"
                )
            ).scalar_one() == "generation_failed"
        engine.dispose()
    finally:
        await _drop_scratch_database(db_name)
