"""Add source identity and content uniqueness for rubric seeding.

Revision ID: 018_dynamic_rubric_seed_identity
Revises: 017_session_report_hardening
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "018_dynamic_rubric_seed_identity"
down_revision: str | None = "017_session_report_hardening"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SOURCE_COLUMNS = (
    sa.Column("source_id", sa.String(length=120), nullable=True),
    sa.Column("source_rubric_key", sa.String(length=120), nullable=True),
)
SOURCE_CONSTRAINT = "uq_negotiation_standards_source_identity"
CONTENT_CONSTRAINT = "uq_negotiation_standard_versions_content_hash"
SOURCE_INDEX = "ix_negotiation_standards_source_identity"


def _constraint_names(bind: sa.Connection, table_name: str) -> set[str]:
    """Return named unique constraints currently defined on a table."""
    return {
        constraint["name"]
        for constraint in sa.inspect(bind).get_unique_constraints(table_name)
        if constraint.get("name")
    }


def _index_names(bind: sa.Connection, table_name: str) -> set[str]:
    """Return named indexes currently defined on a table."""
    return {
        index["name"] for index in sa.inspect(bind).get_indexes(table_name) if index.get("name")
    }


def upgrade() -> None:
    """Add nullable source identity fields and database-enforced deduplication."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    standard_columns = {column["name"] for column in inspector.get_columns("negotiation_standards")}
    missing_columns = [column for column in SOURCE_COLUMNS if column.name not in standard_columns]
    constraint_names = _constraint_names(bind, "negotiation_standards")

    if missing_columns or SOURCE_CONSTRAINT not in constraint_names:
        with op.batch_alter_table("negotiation_standards") as batch_op:
            for column in missing_columns:
                batch_op.add_column(column)
            if SOURCE_CONSTRAINT not in constraint_names:
                batch_op.create_unique_constraint(
                    SOURCE_CONSTRAINT,
                    ["source_id", "source_rubric_key"],
                )

    bind = op.get_bind()
    if SOURCE_INDEX not in _index_names(bind, "negotiation_standards"):
        op.create_index(
            SOURCE_INDEX,
            "negotiation_standards",
            ["source_id", "source_rubric_key"],
        )

    bind = op.get_bind()
    if CONTENT_CONSTRAINT not in _constraint_names(bind, "negotiation_standard_versions"):
        with op.batch_alter_table("negotiation_standard_versions") as batch_op:
            batch_op.create_unique_constraint(
                CONTENT_CONSTRAINT,
                ["standard_id", "content_hash"],
            )


def downgrade() -> None:
    """Remove seed-specific constraints and nullable identity fields."""
    bind = op.get_bind()
    if SOURCE_INDEX in _index_names(bind, "negotiation_standards"):
        op.drop_index(SOURCE_INDEX, table_name="negotiation_standards")

    bind = op.get_bind()
    if CONTENT_CONSTRAINT in _constraint_names(bind, "negotiation_standard_versions"):
        with op.batch_alter_table("negotiation_standard_versions") as batch_op:
            batch_op.drop_constraint(CONTENT_CONSTRAINT, type_="unique")

    bind = op.get_bind()
    standard_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("negotiation_standards")
    }
    constraint_names = _constraint_names(bind, "negotiation_standards")
    if SOURCE_CONSTRAINT in constraint_names or any(
        column.name in standard_columns for column in SOURCE_COLUMNS
    ):
        with op.batch_alter_table("negotiation_standards") as batch_op:
            if SOURCE_CONSTRAINT in constraint_names:
                batch_op.drop_constraint(SOURCE_CONSTRAINT, type_="unique")
            for column in reversed(SOURCE_COLUMNS):
                if column.name in standard_columns:
                    batch_op.drop_column(column.name)
