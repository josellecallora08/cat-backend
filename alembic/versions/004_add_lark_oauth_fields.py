"""Add Lark OAuth fields to users table

Revision ID: 004_add_lark_oauth_fields
Revises: 003_add_password_reset
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004_add_lark_oauth_fields"
down_revision: Union[str, None] = "003_add_password_reset"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("users")}

    # Make hashed_password nullable for OAuth-only users
    if "hashed_password" in columns:
        op.alter_column(
            "users",
            "hashed_password",
            existing_type=sa.String(255),
            nullable=True,
        )

    # Add auth_provider column
    if "auth_provider" not in columns:
        op.add_column(
            "users",
            sa.Column(
                "auth_provider",
                sa.String(length=20),
                nullable=False,
                server_default="local",
            ),
        )

    # Add lark_open_id column
    if "lark_open_id" not in columns:
        op.add_column(
            "users",
            sa.Column("lark_open_id", sa.String(length=255), nullable=True),
        )
        op.create_index("ix_users_lark_open_id", "users", ["lark_open_id"], unique=True)

    # Add lark_union_id column
    if "lark_union_id" not in columns:
        op.add_column(
            "users",
            sa.Column("lark_union_id", sa.String(length=255), nullable=True),
        )
        op.create_index(
            "ix_users_lark_union_id", "users", ["lark_union_id"], unique=True
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("users")}

    if "lark_union_id" in columns:
        op.drop_index("ix_users_lark_union_id", table_name="users")
        op.drop_column("users", "lark_union_id")

    if "lark_open_id" in columns:
        op.drop_index("ix_users_lark_open_id", table_name="users")
        op.drop_column("users", "lark_open_id")

    if "auth_provider" in columns:
        op.drop_column("users", "auth_provider")

    if "hashed_password" in columns:
        op.alter_column(
            "users",
            "hashed_password",
            existing_type=sa.String(255),
            nullable=False,
        )
