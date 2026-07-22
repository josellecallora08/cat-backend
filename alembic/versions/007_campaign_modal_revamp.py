"""Make campaign dates nullable, add role to campaign_agents

Revision ID: 007_campaign_modal_revamp
Revises: 006_add_campaigns
Create Date: 2025-01-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007_campaign_modal_revamp"
down_revision: Union[str, None] = "006_add_campaigns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make start_date and end_date nullable
    op.alter_column("campaigns", "start_date", existing_type=sa.Date(), nullable=True)
    op.alter_column("campaigns", "end_date", existing_type=sa.Date(), nullable=True)

    # Add role column to campaign_agents
    op.add_column(
        "campaign_agents",
        sa.Column(
            "role",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'participant'"),
        ),
    )


def downgrade() -> None:
    # Drop role column from campaign_agents
    op.drop_column("campaign_agents", "role")

    # Backfill any null dates with current date before restoring NOT NULL
    op.execute(
        "UPDATE campaigns SET start_date = CURRENT_DATE WHERE start_date IS NULL"
    )
    op.execute("UPDATE campaigns SET end_date = CURRENT_DATE WHERE end_date IS NULL")

    # Restore NOT NULL constraints on date columns
    op.alter_column("campaigns", "start_date", existing_type=sa.Date(), nullable=False)
    op.alter_column("campaigns", "end_date", existing_type=sa.Date(), nullable=False)
