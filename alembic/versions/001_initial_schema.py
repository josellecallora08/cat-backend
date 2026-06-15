"""Initial schema for Collection Agent Trainer

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- scenarios ---
    op.create_table(
        "scenarios",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("scenario_type", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("debtor_profile", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
    )
    op.create_index("ix_scenarios_is_active", "scenarios", ["is_active"])

    # --- sessions ---
    op.create_table(
        "sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scenario_id", sa.Uuid(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status", sa.String(length=20), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column("persona_context", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["scenario_id"], ["scenarios.id"]),
    )

    # --- transcripts ---
    op.create_table(
        "transcripts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("speaker", sa.String(length=10), nullable=False),
        sa.Column("utterance_text", sa.Text(), nullable=False),
        sa.Column("timestamp_ms", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index("ix_transcripts_session_id", "transcripts", ["session_id"])

    # --- evaluations ---
    op.create_table(
        "evaluations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("category_scores", sa.JSON(), nullable=False),
        sa.Column("strengths", sa.JSON(), nullable=False),
        sa.Column("weaknesses", sa.JSON(), nullable=False),
        sa.Column("is_too_short", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index("ix_evaluations_session_id", "evaluations", ["session_id"])

    # --- coaching_reports ---
    op.create_table(
        "coaching_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("mistakes_by_category", sa.JSON(), nullable=False),
        sa.Column("total_mistakes", sa.Integer(), nullable=False),
        sa.Column("no_mistakes", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index("ix_coaching_reports_session_id", "coaching_reports", ["session_id"])

    # --- learning_plans ---
    op.create_table(
        "learning_plans",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.Column("weak_competencies", sa.JSON(), nullable=False),
        sa.Column("all_passing", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index("ix_learning_plans_session_id", "learning_plans", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_learning_plans_session_id", table_name="learning_plans")
    op.drop_table("learning_plans")

    op.drop_index("ix_coaching_reports_session_id", table_name="coaching_reports")
    op.drop_table("coaching_reports")

    op.drop_index("ix_evaluations_session_id", table_name="evaluations")
    op.drop_table("evaluations")

    op.drop_index("ix_transcripts_session_id", table_name="transcripts")
    op.drop_table("transcripts")

    op.drop_table("sessions")

    op.drop_index("ix_scenarios_is_active", table_name="scenarios")
    op.drop_table("scenarios")
