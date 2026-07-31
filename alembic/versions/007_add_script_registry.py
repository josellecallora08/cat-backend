"""Add scripts, script_versions tables and sessions.script_version_id

Revision ID: 007_add_script_registry
Revises: 006_add_campaigns
Create Date: 2025-01-02 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "007_add_script_registry"
down_revision: Union[str, None] = "006_add_campaigns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    json_variant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

    # --- scripts ---
    if "scripts" not in existing_tables:
        op.create_table(
            "scripts",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("scenario_id", sa.Uuid(), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column(
                "status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'draft'"),
            ),
            sa.Column("format", sa.String(length=10), nullable=False),
            sa.Column("draft_content", json_variant, nullable=True),
            sa.Column("current_version_id", sa.Uuid(), nullable=True),
            sa.Column(
                "is_deleted",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
            sa.Column("created_by", sa.Uuid(), nullable=False),
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
            sa.UniqueConstraint("scenario_id"),
            sa.ForeignKeyConstraint(
                ["scenario_id"],
                ["scenarios.id"],
            ),
            sa.ForeignKeyConstraint(
                ["created_by"],
                ["users.id"],
            ),
        )

    # --- script_versions ---
    if "script_versions" not in existing_tables:
        op.create_table(
            "script_versions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("script_id", sa.Uuid(), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("content", json_variant, nullable=False),
            sa.Column("published_by", sa.Uuid(), nullable=False),
            sa.Column(
                "published_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("script_id", "version_number"),
            sa.ForeignKeyConstraint(
                ["script_id"],
                ["scripts.id"],
            ),
            sa.ForeignKeyConstraint(
                ["published_by"],
                ["users.id"],
            ),
        )

    # --- scripts.current_version_id -> script_versions.id ---
    # Added after script_versions exists to satisfy the FK target.
    existing_fks = {
        fk["name"] for fk in inspector.get_foreign_keys("scripts")
    } if "scripts" in existing_tables else set()
    if "fk_scripts_current_version_id_script_versions" not in existing_fks:
        with op.batch_alter_table("scripts") as batch_op:
            batch_op.create_foreign_key(
                "fk_scripts_current_version_id_script_versions",
                "script_versions",
                ["current_version_id"],
                ["id"],
            )

    # --- sessions.script_version_id ---
    existing_session_columns = {
        col["name"] for col in inspector.get_columns("sessions")
    }
    if "script_version_id" not in existing_session_columns:
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.add_column(
                sa.Column("script_version_id", sa.Uuid(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_sessions_script_version_id_script_versions",
                "script_versions",
                ["script_version_id"],
                ["id"],
            )


def downgrade() -> None:
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_constraint(
            "fk_sessions_script_version_id_script_versions",
            type_="foreignkey",
        )
        batch_op.drop_column("script_version_id")

    with op.batch_alter_table("scripts") as batch_op:
        batch_op.drop_constraint(
            "fk_scripts_current_version_id_script_versions",
            type_="foreignkey",
        )

    op.drop_table("script_versions")
    op.drop_table("scripts")
