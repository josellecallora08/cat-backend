"""Add campaigns, campaign_scenarios, and campaign_agents tables

Revision ID: 006_add_campaigns
Revises: 005_add_google_oauth_field
Create Date: 2025-01-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006_add_campaigns"
down_revision: Union[str, None] = "005_add_google_oauth_field"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # --- campaigns ---
    if "campaigns" not in existing_tables:
        op.create_table(
            "campaigns",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'draft'"),
            ),
            sa.Column("start_date", sa.Date(), nullable=False),
            sa.Column("end_date", sa.Date(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )

    # --- campaign_scenarios ---
    if "campaign_scenarios" not in existing_tables:
        op.create_table(
            "campaign_scenarios",
            sa.Column("campaign_id", sa.Uuid(), nullable=False),
            sa.Column("scenario_id", sa.Uuid(), nullable=False),
            sa.PrimaryKeyConstraint("campaign_id", "scenario_id"),
            sa.ForeignKeyConstraint(
                ["campaign_id"],
                ["campaigns.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["scenario_id"],
                ["scenarios.id"],
                ondelete="CASCADE",
            ),
        )

    # --- campaign_agents ---
    if "campaign_agents" not in existing_tables:
        op.create_table(
            "campaign_agents",
            sa.Column("campaign_id", sa.Uuid(), nullable=False),
            sa.Column("agent_id", sa.Uuid(), nullable=False),
            sa.PrimaryKeyConstraint("campaign_id", "agent_id"),
            sa.ForeignKeyConstraint(
                ["campaign_id"],
                ["campaigns.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["agent_id"],
                ["users.id"],
                ondelete="CASCADE",
            ),
        )


def downgrade() -> None:
    op.drop_table("campaign_agents")
    op.drop_table("campaign_scenarios")
    op.drop_table("campaigns")
