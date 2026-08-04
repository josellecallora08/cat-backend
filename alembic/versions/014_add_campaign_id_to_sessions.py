"""Add nullable campaign linkage to sessions.

Revision ID: 014_add_campaign_id_to_sessions
Revises: 013_merge_heads
Create Date: 2026-08-01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "014_add_campaign_id_to_sessions"
down_revision: Union[str, None] = "013_merge_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the nullable campaign foreign key and supporting index."""
    op.add_column("sessions", sa.Column("campaign_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_sessions_campaign_id",
        "sessions",
        "campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_sessions_campaign_id", "sessions", ["campaign_id"])


def downgrade() -> None:
    """Remove the campaign foreign key, index, and column."""
    op.drop_index("ix_sessions_campaign_id", table_name="sessions")
    op.drop_constraint("fk_sessions_campaign_id", "sessions", type_="foreignkey")
    op.drop_column("sessions", "campaign_id")
