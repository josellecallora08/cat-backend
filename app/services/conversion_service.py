"""S1-09: Upload-to-Script conversion orchestration service.

Owns the full atomic conversion workflow:
- Row locking on the ScriptUpload
- Eligibility checks
- Scenario validation
- Content conversion and full limit validation
- Transaction-aware draft creation (flush)
- Upload linkage
- Single commit
- IntegrityError classification
- Rollback on failure

The FastAPI endpoint delegates to this service and translates outcomes to HTTP.
"""

import json
import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.script import Script
from app.models.script_upload import ScriptUpload, UploadStatus
from app.services.script_converter import ConversionError, convert_extracted_to_contract
from app.services.script_registry import create_draft_in_transaction
from app.services.script_validator import (
    ScriptFormatError,
    ScriptLimits,
    ScriptValidationError,
    validate_script,
)


logger = logging.getLogger(__name__)


# The exact constraint name PostgreSQL generates for
# Column("scenario_id", ..., unique=True) on __tablename__ = "scripts"
_SCRIPT_SCENARIO_CONSTRAINT = "scripts_scenario_id_key"


def is_scenario_script_unique_violation(exc: IntegrityError) -> bool:
    """Classify whether an IntegrityError is the scripts.scenario_id unique violation.

    Checks SQLSTATE 23505 (unique_violation) + the exact constraint name:
        scripts_scenario_id_key

    Uses structured metadata first, then a safe message-parsing fallback that
    extracts the quoted constraint identifier and compares with exact equality.
    Does NOT use substring matching — a constraint named
    'other_scripts_scenario_id_key' or 'scripts_scenario_id_key_backup'
    will correctly return False.
    """
    orig = exc.orig
    if orig is None:
        return False

    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    if sqlstate != "23505":
        return False

    # 1. asyncpg structured field (preferred)
    constraint = getattr(orig, "constraint_name", None)
    if constraint == _SCRIPT_SCENARIO_CONSTRAINT:
        return True
    if constraint is not None:
        # Structured metadata present but does not match — definitive answer
        return False

    # 2. psycopg2/psycopg diag field
    if hasattr(orig, "diag") and hasattr(orig.diag, "constraint_name"):
        diag_constraint = orig.diag.constraint_name
        if diag_constraint == _SCRIPT_SCENARIO_CONSTRAINT:
            return True
        if diag_constraint is not None:
            return False

    # 3. Message-only fallback: parse the quoted constraint identifier
    #    PostgreSQL format: ... unique constraint "constraint_name"
    import re
    orig_msg = str(orig)
    match = re.search(r'"([^"]+)"', orig_msg)
    if match:
        parsed_constraint = match.group(1)
        return parsed_constraint == _SCRIPT_SCENARIO_CONSTRAINT

    # Cannot determine constraint unambiguously — do not guess
    return False


# --- Result types ---


@dataclass
class ConversionSuccess:
    """Successful conversion result."""
    upload_id: UUID
    script_id: UUID
    scenario_id: UUID


@dataclass
class ConversionConflict:
    """Conflict: upload already converted or scenario already has a script."""
    error: str  # "already_converted" or "scenario_has_script"
    message: str
    existing_script_id: UUID | None = None


@dataclass
class ConversionRejection:
    """Eligibility or validation rejection."""
    error: str  # "upload_ineligible" or "conversion_failed"
    message: str
    details: dict | None = None


class ConversionInternalError(Exception):
    """Unrecoverable internal error during conversion."""


ConversionResult = ConversionSuccess | ConversionConflict | ConversionRejection


async def convert_upload_to_script_draft(
    db: AsyncSession,
    upload_id: UUID,
    admin_id: UUID,
) -> ConversionResult:
    """Execute the full upload-to-script conversion atomically.

    Uses SELECT ... FOR UPDATE to lock the upload row, validates eligibility,
    converts content, runs full configurable validation, creates a draft via
    flush, links the upload, and commits once.

    On failure, rollback removes both the draft and linkage.

    Returns:
        ConversionSuccess, ConversionConflict, or ConversionRejection.

    Raises:
        ConversionInternalError: Unrecoverable DB error after rollback.
    """
    # 1. Load and lock upload row
    stmt = (
        select(ScriptUpload)
        .where(ScriptUpload.id == upload_id)
        .with_for_update()
    )
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()

    if upload is None:
        return ConversionRejection(
            error="not_found", message="Upload not found"
        )

    # 2. Duplicate conversion check (under lock)
    if upload.script_id is not None:
        return ConversionConflict(
            error="already_converted",
            message="This upload has already been converted to a script.",
            existing_script_id=upload.script_id,
        )

    # 3. Eligibility checks
    rejection = _check_eligibility(upload)
    if rejection is not None:
        return rejection

    # 4. Validate scenario exists
    from app.models import Scenario
    scenario_result = await db.execute(
        select(Scenario).where(Scenario.id == upload.scenario_id)
    )
    if scenario_result.scalar_one_or_none() is None:
        return ConversionRejection(
            error="upload_ineligible",
            message=f"Scenario '{upload.scenario_id}' does not exist.",
        )

    # 5. Check existing Script for this scenario
    existing_script_result = await db.execute(
        select(Script).where(
            Script.scenario_id == upload.scenario_id,
            Script.is_deleted == False,  # noqa: E712
        )
    )
    existing_script = existing_script_result.scalar_one_or_none()
    if existing_script is not None:
        return ConversionConflict(
            error="scenario_has_script",
            message="This scenario already has a script.",
            existing_script_id=existing_script.id,
        )

    # 6. Convert extracted_content
    try:
        contract_data = convert_extracted_to_contract(upload.extracted_content)
    except ConversionError as exc:
        return ConversionRejection(
            error="conversion_failed",
            message=str(exc),
            details=exc.details,
        )

    # 7. Full configurable limit validation
    raw_definition = json.dumps(contract_data)
    limits = ScriptLimits(
        max_definition_size_bytes=settings.script_max_definition_size_bytes,
        max_trigger_phrases=settings.script_max_trigger_phrases,
        max_expected_replies=settings.script_max_expected_replies,
        max_escalation_conditions=settings.script_max_escalation_conditions,
        max_field_text_length=settings.script_max_field_text_length,
    )
    try:
        validate_script(raw_definition, "json", limits)
    except ScriptFormatError as exc:
        return ConversionRejection(
            error="conversion_failed",
            message=f"Converted contract failed format validation: {exc}",
        )
    except ScriptValidationError as exc:
        return ConversionRejection(
            error="conversion_failed",
            message=f"Converted contract does not meet publication requirements ({len(exc.errors)} issue(s)).",
            details={"validation_errors": exc.errors},
        )

    # 8. Create draft via flush (no commit yet)
    try:
        script = await create_draft_in_transaction(
            db,
            admin_id=admin_id,
            name=f"Converted from upload {upload.filename_original}",
            scenario_id=upload.scenario_id,
            format="json",
            raw_definition=raw_definition,
        )
    except (ScriptFormatError, ScriptValidationError) as exc:
        await db.rollback()
        return ConversionRejection(
            error="conversion_failed",
            message=f"Script registry validation failed: {exc}",
            details=getattr(exc, "errors", None),
        )
    except IntegrityError as exc:
        await db.rollback()
        if is_scenario_script_unique_violation(exc):
            return ConversionConflict(
                error="scenario_has_script",
                message="This scenario already has a script.",
            )
        logger.error(
            "conversion_integrity_error",
            extra={
                "upload_id": str(upload_id),
                "scenario_id": str(upload.scenario_id),
                "user_id": str(admin_id),
                "category": "unknown_integrity_error",
            },
        )
        raise ConversionInternalError("Unexpected integrity constraint violation")
    except Exception as exc:
        await db.rollback()
        logger.error(
            "conversion_registry_failure",
            extra={
                "upload_id": str(upload_id),
                "user_id": str(admin_id),
                "error": type(exc).__name__,
            },
        )
        raise ConversionInternalError("Failed to create script draft")

    # 9. Link upload
    upload.script_id = script.id

    # 10. Single atomic commit
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        if is_scenario_script_unique_violation(exc):
            return ConversionConflict(
                error="scenario_has_script",
                message="This scenario already has a script.",
            )
        logger.error(
            "conversion_commit_integrity_error",
            extra={
                "upload_id": str(upload_id),
                "scenario_id": str(upload.scenario_id),
                "user_id": str(admin_id),
                "category": "unknown_integrity_error",
            },
        )
        raise ConversionInternalError("Unexpected constraint violation on commit")
    except Exception as exc:
        await db.rollback()
        logger.error(
            "conversion_commit_failure",
            extra={
                "upload_id": str(upload_id),
                "user_id": str(admin_id),
                "error": type(exc).__name__,
            },
        )
        raise ConversionInternalError("Failed to commit conversion")

    # 11. Audit log — only after successful commit
    logger.info(
        "upload_converted",
        extra={
            "upload_id": str(upload_id),
            "script_id": str(script.id),
            "scenario_id": str(upload.scenario_id),
            "user_id": str(admin_id),
            "format": "json",
        },
    )

    return ConversionSuccess(
        upload_id=upload.id,
        script_id=script.id,
        scenario_id=upload.scenario_id,
    )


def _check_eligibility(upload: ScriptUpload) -> ConversionRejection | None:
    """Check upload eligibility. Returns a rejection or None if eligible."""
    if upload.status == UploadStatus.DELETED.value:
        return ConversionRejection(error="upload_ineligible", message="Upload has been deleted.")
    if upload.scan_status == "infected":
        return ConversionRejection(error="upload_ineligible", message="Upload is infected and cannot be converted.")
    if upload.scan_status == "error":
        return ConversionRejection(error="upload_ineligible", message="Upload scan failed and cannot be converted.")
    if upload.scan_status == "pending":
        return ConversionRejection(error="upload_ineligible", message="Upload scan is still pending.")
    if upload.scan_status != "clean":
        return ConversionRejection(error="upload_ineligible", message=f"Upload scan status '{upload.scan_status}' is not eligible for conversion.")
    if upload.extraction_status != "completed":
        return ConversionRejection(error="upload_ineligible", message=f"Extraction status '{upload.extraction_status}' is not eligible for conversion.")
    if upload.status != UploadStatus.COMPLETED.value:
        return ConversionRejection(error="upload_ineligible", message=f"Upload status '{upload.status}' is not eligible for conversion. Must be 'completed'.")
    if not upload.extracted_content or not upload.extracted_content.strip():
        return ConversionRejection(error="upload_ineligible", message="Upload has no extracted content available.")
    if upload.scenario_id is None:
        return ConversionRejection(error="upload_ineligible", message="Upload has no scenario_id. A valid scenario is required for conversion.")
    return None
