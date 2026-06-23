"""Add password_reset_tokens table and password_changed_at column.

Revision ID: 003
Revises: 002
Create Date: 2026-06-22
"""

from alembic import op
import sqlalchemy as sa

revision = "003_add_password_reset"
down_revision = "002_add_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add password_changed_at to users
    op.add_column("users", sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True))

    # Create password_reset_tokens table
    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"])
    op.create_index("ix_password_reset_tokens_token_hash", "password_reset_tokens", ["token_hash"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_password_reset_tokens_token_hash", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_user_id", table_name="password_reset_tokens")
    op.drop_table("password_reset_tokens")
    op.drop_column("users", "password_changed_at")
