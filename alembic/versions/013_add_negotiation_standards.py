"""Add campaign-linked negotiation standards and immutable versions.

Revision ID: 013_add_negotiation_standards
Revises: 012_add_upload_rejection_fields
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "013_add_negotiation_standards"
down_revision: Union[str, Sequence[str], None] = (
    "012_add_upload_rejection_fields",
    "010_add_trainer_campaign_role",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json_variant() -> sa.types.TypeEngine:
    """Return the repository's cross-database JSON type."""
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    """Create standard/version tables and nullable historical columns."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "negotiation_standards" not in tables:
        op.create_table(
            "negotiation_standards",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("campaign_id", sa.Uuid(), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("description", sa.String(length=1000), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="draft"),
            sa.Column("overall_passing_score", sa.Integer(), nullable=False, server_default="70"),
            sa.Column("draft_content", _json_variant(), nullable=True),
            sa.Column("current_version_id", sa.Uuid(), nullable=True),
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_by", sa.Uuid(), nullable=False),
            sa.Column("updated_by", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("campaign_id"),
            sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"]),
            sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
            sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
        )
        op.create_index("ix_negotiation_standards_campaign_id", "negotiation_standards", ["campaign_id"])
        op.create_index("ix_negotiation_standards_status", "negotiation_standards", ["status"])

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "negotiation_standard_versions" not in tables:
        op.create_table(
            "negotiation_standard_versions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("standard_id", sa.Uuid(), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("snapshot", _json_variant(), nullable=False),
            sa.Column("content_hash", sa.String(length=64), nullable=False),
            sa.Column("created_by", sa.Uuid(), nullable=False),
            sa.Column("published_by", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("published_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("publication_note", sa.String(length=500), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("standard_id", "version_number"),
            sa.ForeignKeyConstraint(["standard_id"], ["negotiation_standards.id"]),
            sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
            sa.ForeignKeyConstraint(["published_by"], ["users.id"]),
        )
        op.create_index(
            "ix_negotiation_standard_versions_standard_id",
            "negotiation_standard_versions",
            ["standard_id"],
        )
        op.create_index(
            "ix_negotiation_standard_versions_published_at",
            "negotiation_standard_versions",
            ["published_at"],
        )

    standard_fks = {
        fk["name"] for fk in sa.inspect(bind).get_foreign_keys("negotiation_standards")
    }
    if "fk_negotiation_standards_current_version" not in standard_fks:
        with op.batch_alter_table("negotiation_standards") as batch_op:
            batch_op.create_foreign_key(
                "fk_negotiation_standards_current_version",
                "negotiation_standard_versions",
                ["current_version_id"],
                ["id"],
            )

    session_columns = {column["name"] for column in sa.inspect(bind).get_columns("sessions")}
    if "negotiation_standard_version_id" not in session_columns:
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.add_column(sa.Column("negotiation_standard_version_id", sa.Uuid(), nullable=True))
            batch_op.create_foreign_key(
                "fk_sessions_negotiation_standard_version",
                "negotiation_standard_versions",
                ["negotiation_standard_version_id"],
                ["id"],
                ondelete="RESTRICT",
            )

    evaluation_columns = {column["name"] for column in sa.inspect(bind).get_columns("evaluations")}
    additions = {
        "negotiation_standard_version_id": sa.Column("negotiation_standard_version_id", sa.Uuid(), nullable=True),
        "standard_snapshot": sa.Column("standard_snapshot", _json_variant(), nullable=True),
        "weighted_total": sa.Column("weighted_total", sa.Float(), nullable=True),
        "passing_score": sa.Column("passing_score", sa.Integer(), nullable=True),
        "passed": sa.Column("passed", sa.Boolean(), nullable=True),
        "rubric_result": sa.Column("rubric_result", _json_variant(), nullable=True),
    }
    missing = [column for name, column in additions.items() if name not in evaluation_columns]
    if missing:
        with op.batch_alter_table("evaluations") as batch_op:
            for column in missing:
                batch_op.add_column(column)
            if "negotiation_standard_version_id" not in evaluation_columns:
                batch_op.create_foreign_key(
                    "fk_evaluations_negotiation_standard_version",
                    "negotiation_standard_versions",
                    ["negotiation_standard_version_id"],
                    ["id"],
                    ondelete="RESTRICT",
                )


def downgrade() -> None:
    """Remove standards additions while retaining legacy session data."""
    with op.batch_alter_table("evaluations") as batch_op:
        batch_op.drop_constraint(
            "fk_evaluations_negotiation_standard_version", type_="foreignkey"
        )
        for column in (
            "rubric_result",
            "passed",
            "passing_score",
            "weighted_total",
            "standard_snapshot",
            "negotiation_standard_version_id",
        ):
            batch_op.drop_column(column)

    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_constraint(
            "fk_sessions_negotiation_standard_version", type_="foreignkey"
        )
        batch_op.drop_column("negotiation_standard_version_id")

    with op.batch_alter_table("negotiation_standards") as batch_op:
        batch_op.drop_constraint(
            "fk_negotiation_standards_current_version", type_="foreignkey"
        )
    op.drop_index("ix_negotiation_standard_versions_published_at", table_name="negotiation_standard_versions")
    op.drop_index("ix_negotiation_standard_versions_standard_id", table_name="negotiation_standard_versions")
    op.drop_table("negotiation_standard_versions")
    op.drop_index("ix_negotiation_standards_status", table_name="negotiation_standards")
    op.drop_index("ix_negotiation_standards_campaign_id", table_name="negotiation_standards")
    op.drop_table("negotiation_standards")
