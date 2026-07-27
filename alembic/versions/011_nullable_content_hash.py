"""Make script_uploads.content_hash nullable for failed pre-extraction records.

Revision ID: 011_nullable_content_hash
Revises: 010_merge_upload_and_roles
Create Date: 2026-07-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "011_nullable_content_hash"
down_revision: Union[str, None] = "010_merge_upload_and_roles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "script_uploads", "content_hash",
        existing_type=sa.String(64),
        nullable=True,
    )
    # Restore NULL from the empty-string sentinel used by downgrade.
    # Only affects failed pre-extraction records that were backfilled during
    # a previous downgrade cycle. Valid hashes are never empty strings.
    op.execute(
        "UPDATE script_uploads SET content_hash = NULL "
        "WHERE content_hash = '' AND status = 'failed'"
    )


def downgrade() -> None:
    # The old schema requires content_hash NOT NULL. Use empty string as a
    # temporary compatibility sentinel for failed pre-extraction records
    # whose content_hash is legitimately NULL. This sentinel is converted
    # back to NULL by the upgrade path above.
    op.execute(
        "UPDATE script_uploads SET content_hash = '' WHERE content_hash IS NULL"
    )
    op.alter_column(
        "script_uploads", "content_hash",
        existing_type=sa.String(64),
        nullable=False,
    )
