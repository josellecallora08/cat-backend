"""Add an idempotency key to session creation.

Revision ID: 019_session_creation_idempotency
Revises: 018_dynamic_rubric_seed_identity
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "019_session_creation_idempotency"
down_revision: str | None = "018_dynamic_rubric_seed_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("creation_key", sa.Uuid(), nullable=True))
    op.create_unique_constraint(
        "uq_sessions_agent_creation_key",
        "sessions",
        ["agent_id", "creation_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_sessions_agent_creation_key",
        "sessions",
        type_="unique",
    )
    op.drop_column("sessions", "creation_key")
