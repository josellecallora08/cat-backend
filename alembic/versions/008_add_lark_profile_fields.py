"""Add Lark profile fields (avatar_url, employee_id, department)

Revision ID: 008_add_lark_profile_fields
Revises: 007_campaign_modal_revamp
Create Date: 2026-07-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "008_add_lark_profile_fields"
down_revision: Union[str, None] = "007_campaign_modal_revamp"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("avatar_url", sa.String(512), nullable=True))
    op.add_column("users", sa.Column("employee_id", sa.String(100), nullable=True))
    op.add_column("users", sa.Column("department", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "department")
    op.drop_column("users", "employee_id")
    op.drop_column("users", "avatar_url")
