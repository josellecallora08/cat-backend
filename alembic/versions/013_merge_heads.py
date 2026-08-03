"""Merge heads: 010_add_trainer_campaign_role + 012_add_upload_rejection_fields

Revision ID: 013_merge_heads
Revises: 010_add_trainer_campaign_role, 012_add_upload_rejection_fields
Create Date: 2026-07-31

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "013_merge_heads"
down_revision: Union[str, None] = (
    "010_add_trainer_campaign_role",
    "012_add_upload_rejection_fields",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge-only migration, no schema changes."""


def downgrade() -> None:
    """Merge-only migration, no schema changes."""
