"""Add trainer campaign role to CampaignRole enum.

No schema change required — campaign_agents.role is String(20) and the enum
is enforced at the application layer only. This migration exists as an audit
trail documenting the addition of the 'trainer' role value.

Revision ID: 010_add_trainer_campaign_role
Revises: 009_refactor_roles_add_user_type
Create Date: 2026-07-25
"""

from typing import Sequence, Union


revision: str = "010_add_trainer_campaign_role"
down_revision: Union[str, None] = "009_refactor_roles_add_user_type"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: trainer role is enforced at application layer via CampaignRole enum."""
    pass


def downgrade() -> None:
    """No-op: removing the enum value from code is sufficient to disallow the role."""
    pass
