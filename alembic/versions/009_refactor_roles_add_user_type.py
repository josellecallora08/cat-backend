"""Refactor roles: add user_type column, migrate agent role to user+agent

Revision ID: 009_refactor_roles_add_user_type
Revises: 008_add_lark_profile_fields
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "009_refactor_roles_add_user_type"
down_revision: Union[str, None] = "008_add_lark_profile_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("user_type", sa.String(20), nullable=True))
    op.execute(
        "UPDATE users SET user_type = 'agent', role = 'user' WHERE role = 'agent'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE users SET role = 'agent' WHERE role = 'user' AND user_type = 'agent'"
    )
    op.drop_column("users", "user_type")
