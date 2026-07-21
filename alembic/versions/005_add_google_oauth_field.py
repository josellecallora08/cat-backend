"""Add google_sub field to users table

Revision ID: 005_add_google_oauth_field
Revises: 004_add_lark_oauth_fields
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005_add_google_oauth_field"
down_revision: Union[str, None] = "004_add_lark_oauth_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("users")}

    if "google_sub" not in columns:
        op.add_column(
            "users",
            sa.Column("google_sub", sa.String(length=255), nullable=True),
        )
        op.create_index("ix_users_google_sub", "users", ["google_sub"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("users")}

    if "google_sub" in columns:
        op.drop_index("ix_users_google_sub", table_name="users")
        op.drop_column("users", "google_sub")
