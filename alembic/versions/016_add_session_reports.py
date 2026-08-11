"""Add session_reports table for versioned session report snapshots.

Revision ID: 016_add_session_reports
Revises: 015_merge_campaign_negotiation
Create Date: 2026-08-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "016_add_session_reports"
down_revision: Union[str, None] = "015_merge_campaign_negotiation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the session_reports table, constraints, and indexes."""
    op.create_table(
        "session_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.Column("campaign_id", sa.Uuid(), nullable=True),
        sa.Column("negotiation_standard_version_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("report_version", sa.Integer(), nullable=False),
        sa.Column(
            "payload",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("generated_by", sa.Uuid(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_session_reports_session_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["campaign_id"],
            ["campaigns.id"],
            name="fk_session_reports_campaign_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["negotiation_standard_version_id"],
            ["negotiation_standard_versions.id"],
            name="fk_session_reports_standard_version_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "session_id", "report_version", name="uq_session_reports_session_version"
        ),
    )
    op.create_index(
        "ix_session_reports_session_id_version_desc",
        "session_reports",
        ["session_id", sa.text("report_version DESC")],
    )
    op.create_index(
        "ix_session_reports_campaign_id", "session_reports", ["campaign_id"]
    )


def downgrade() -> None:
    """Drop the session_reports table and its indexes."""
    op.drop_index("ix_session_reports_campaign_id", table_name="session_reports")
    op.drop_index(
        "ix_session_reports_session_id_version_desc", table_name="session_reports"
    )
    op.drop_table("session_reports")
