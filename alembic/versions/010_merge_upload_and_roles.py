"""Merge heads: 008_add_script_uploads + 009_refactor_roles_add_user_type

Revision ID: 010_merge_upload_and_roles
Revises: 008_add_script_uploads, 009_refactor_roles_add_user_type
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op


revision: str = "010_merge_upload_and_roles"
down_revision: Union[str, Sequence[str], None] = (
    "008_add_script_uploads",
    "009_refactor_roles_add_user_type",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
