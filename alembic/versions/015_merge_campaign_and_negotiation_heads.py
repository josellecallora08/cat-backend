"""Merge campaign-session and negotiation-standard migration heads.

Revision ID: 015_merge_campaign_negotiation
Revises: 014_add_campaign_id_to_sessions, 014_negotiation_linear
"""

from typing import Sequence, Union


revision: str = "015_merge_campaign_negotiation"
down_revision: Union[str, Sequence[str], None] = (
    "014_add_campaign_id_to_sessions",
    "014_negotiation_linear",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge both migration branches without changing the schema."""


def downgrade() -> None:
    """Return to the two pre-merge migration heads."""
