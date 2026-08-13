"""Keep the repository head linear after the historical migration merge.

Revision ID: 014_negotiation_linear
Revises: 013_add_negotiation_standards
"""

from collections.abc import Sequence


revision: str = "014_negotiation_linear"
down_revision: str | None = "013_merge_heads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Advance to a linear head without changing the schema."""


def downgrade() -> None:
    """Return to the negotiation standards migration."""
