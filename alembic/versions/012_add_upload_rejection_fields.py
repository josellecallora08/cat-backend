"""Add rejection metadata fields to script_uploads for S1-10 admin review.

Revision ID: 012_add_upload_rejection_fields
Revises: 011_nullable_content_hash
Create Date: 2026-07-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "012_add_upload_rejection_fields"
down_revision: Union[str, None] = "011_nullable_content_hash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {col["name"] for col in inspector.get_columns("script_uploads")}

    if "rejected_at" not in existing_columns:
        op.add_column(
            "script_uploads",
            sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
        )

    if "rejected_by" not in existing_columns:
        op.add_column(
            "script_uploads",
            sa.Column("rejected_by", sa.Uuid(), nullable=True),
        )
        op.create_foreign_key(
            "fk_script_uploads_rejected_by_users",
            "script_uploads",
            "users",
            ["rejected_by"],
            ["id"],
        )

    if "rejection_reason" not in existing_columns:
        op.add_column(
            "script_uploads",
            sa.Column("rejection_reason", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    op.drop_constraint(
        "fk_script_uploads_rejected_by_users",
        "script_uploads",
        type_="foreignkey",
    )
    op.drop_column("script_uploads", "rejection_reason")
    op.drop_column("script_uploads", "rejected_by")
    op.drop_column("script_uploads", "rejected_at")
