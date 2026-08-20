"""Merge heads: 010_add_trainer_campaign_role + 012_add_upload_rejection_fields

Revision ID: 013_merge_heads
Revises: 010_add_trainer_campaign_role, 012_add_upload_rejection_fields
Create Date: 2026-07-31

"""

from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "013_merge_heads"
down_revision: str | None = "013_add_negotiation_standards"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Merge-only migration, no schema changes."""


def downgrade() -> None:
    """Merge-only migration, no schema changes."""
