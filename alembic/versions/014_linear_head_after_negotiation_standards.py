"""Keep the repository head linear after the historical migration merge.

Revision ID: 014_negotiation_linear
Revises: 013_add_negotiation_standards
"""

from typing import Sequence, Union

from alembic import op


revision: str = "014_negotiation_linear"
down_revision: Union[str, None] = "013_add_negotiation_standards"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Advance to a linear head without changing the schema."""


def downgrade() -> None:
    """Return to the negotiation standards migration."""
