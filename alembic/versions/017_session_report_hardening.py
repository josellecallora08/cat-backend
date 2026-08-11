"""Harden session report row invariants without rewriting the 016 baseline.

Revision ID: 017_session_report_hardening
Revises: 016_add_session_reports

The 016 revision is already an applied local revision.  This successor adds
only the typed row-level reason and portable persistence checks.  Downgrading
this revision deliberately preserves the original 016 report rows; a further
downgrade through 016 retains the historical table-removal behavior.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "017_session_report_hardening"
down_revision: Union[str, None] = "016_add_session_reports"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLE = "session_reports"
REASON_CODES = (
    "artifact_missing",
    "empty_transcript",
    "not_applicable",
    "session_too_short",
    "generation_pending",
    "generation_failed",
    "legacy_only",
    "no_evidence",
    "no_coaching",
    "no_learning_plan",
)


# Keep each invariant separately named so operators can identify a failed
# contract and so the downgrade can remove only this revision's additions.
CONSTRAINTS = {
    "ck_session_reports_status": (
        "status IN ('pending', 'ready', 'failed')"
    ),
    "ck_session_reports_reason_code": (
        "reason_code IS NULL OR reason_code IN ("
        + ", ".join(f"'{code}'" for code in REASON_CODES)
        + ")"
    ),
    "ck_session_reports_positive_version": "report_version >= 1",
    "ck_session_reports_hash_length": (
        "content_hash IS NULL OR length(content_hash) = 64"
    ),
    "ck_session_reports_status_payload": (
        "(status = 'ready' AND payload IS NOT NULL AND content_hash IS NOT NULL "
        "AND reason_code IS NULL AND failure_reason IS NULL) "
        "OR (status = 'pending' AND payload IS NULL AND content_hash IS NULL "
        "AND reason_code = 'generation_pending') "
        "OR (status = 'failed' AND payload IS NULL AND content_hash IS NULL "
        "AND reason_code = 'generation_failed')"
    ),
}


def upgrade() -> None:
    """Add typed reasons and enforce safe immutable report-row combinations."""
    op.add_column(TABLE, sa.Column("reason_code", sa.String(length=40), nullable=True))

    # 016 had no typed reason column.  Backfill only the states that have a
    # required row-level reason; ready snapshots intentionally remain NULL.
    op.execute(
        sa.text(
            "UPDATE session_reports "
            "SET reason_code = 'generation_pending' "
            "WHERE status = 'pending' AND reason_code IS NULL"
        )
    )
    op.execute(
        sa.text(
            "UPDATE session_reports "
            "SET reason_code = 'generation_failed' "
            "WHERE status = 'failed' AND reason_code IS NULL"
        )
    )

    for name, condition in CONSTRAINTS.items():
        op.create_check_constraint(name, TABLE, condition)


def downgrade() -> None:
    """Remove hardening while preserving every original 016 report row."""
    for name in reversed(tuple(CONSTRAINTS)):
        op.drop_constraint(name, TABLE, type_="check")
    op.drop_column(TABLE, "reason_code")
