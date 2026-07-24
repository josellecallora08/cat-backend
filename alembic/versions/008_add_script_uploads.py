"""Add script_uploads table for upload metadata tracking.

Revision ID: 008_add_script_uploads
Revises: 007_add_script_registry
Create Date: 2025-01-03 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "008_add_script_uploads"
down_revision: Union[str, Sequence[str], None] = ("007_add_script_registry", "007_campaign_modal_revamp")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "script_uploads" not in existing_tables:
        op.create_table(
            "script_uploads",
            sa.Column("id", sa.Uuid(), nullable=False),
            # File identity
            sa.Column("filename_original", sa.String(length=255), nullable=False),
            sa.Column("mime_type", sa.String(length=100), nullable=False),
            sa.Column("file_size_bytes", sa.Integer(), nullable=False),
            sa.Column("content_hash", sa.String(length=64), nullable=False),
            # Storage
            sa.Column("storage_key", sa.String(length=255), nullable=False),
            # Uploader
            sa.Column("uploaded_by", sa.Uuid(), nullable=False),
            # Processing status
            sa.Column(
                "scan_status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'pending'"),
            ),
            sa.Column("scan_signature", sa.String(length=255), nullable=True),
            sa.Column(
                "extraction_status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'pending'"),
            ),
            sa.Column("extraction_error", sa.Text(), nullable=True),
            # Extracted content (pending S1-09 ScriptContract conversion)
            sa.Column("extracted_content", sa.Text(), nullable=True),
            # Scenario link
            sa.Column("scenario_id", sa.Uuid(), nullable=True),
            # Overall status
            sa.Column(
                "status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'pending'"),
            ),
            # Optional link to script
            sa.Column("script_id", sa.Uuid(), nullable=True),
            # Timestamps
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
            sa.Column("quarantine_expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            # Constraints
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(["uploaded_by"], ["users.id"]),
            sa.ForeignKeyConstraint(["script_id"], ["scripts.id"]),
            sa.ForeignKeyConstraint(["scenario_id"], ["scenarios.id"]),
        )


def downgrade() -> None:
    op.drop_table("script_uploads")
