"""S1-10: Admin review service — derives review state and orchestrates actions.

Review status is DERIVED from the combination of:
- ScriptUpload.scan_status
- ScriptUpload.extraction_status
- ScriptUpload.status (UploadStatus enum)
- Script.status (ScriptStatus enum), if a linked script exists

State derivation (order matters — first match wins):
─────────────────────────────────────────────────────
  upload.status == "rejected"                        → "rejected"
  scan_status == "infected"                          → "infected"
  scan_status == "error"                             → "scan_failed"
  extraction_status == "failed"                      → "extraction_failed"
  upload.status == "failed"                          → "processing"
  scan_status == "pending" or "scanning"             → "processing"
  extraction_status == "pending"                     → "processing"
  upload.status in (pending, scanning, extracting)   → "processing"
  upload.status == "deleted"                         → "rejected"
  upload.status == "completed" and script_id is None → "ready_for_conversion"
  script.is_deleted                                  → "rejected"
  script.status == "draft"                           → "ready_for_review"
  script.status == "published"                       → "published"
  script.status == "unpublished"                     → "rejected"
  fallback                                           → "processing"

Valid action transitions (deterministic, server-side only):
  ready_for_review    → edit, reject, publish, retry(conversion)
  ready_for_conversion → retry(conversion), reject
  extraction_failed   → retry(extraction) if source exists, reject
  scan_failed         → retry(scan) if source exists, reject
  infected            → reject
  published           → (none)
  rejected            → (none)
  processing          → (none)
"""

import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.script import Script, ScriptStatus, ScriptVersion
from app.models.script_upload import ScriptUpload, UploadStatus
from app.schemas.review import (
    ReviewDetailResponse,
    ReviewScanResult,
    ReviewWarning,
)
from app.services.script_validator import (
    ScriptFormatError,
    ScriptLimits,
    ScriptValidationError,
    validate_script,
)


logger = logging.getLogger(__name__)


# ─── State derivation ─────────────────────────────────────────────────────────


def derive_review_status(
    upload: ScriptUpload,
    script: Script | None = None,
) -> str:
    """Derive the externally visible review status from persisted state.

    Single authoritative function; endpoints must not scatter ad-hoc state checks.
    """
    # Explicit rejection
    if upload.status == UploadStatus.REJECTED.value:
        return "rejected"

    # Infection / scan failure
    if upload.scan_status == "infected":
        return "infected"
    if upload.scan_status == "error":
        return "scan_failed"

    # Extraction failure
    if upload.extraction_status == "failed":
        return "extraction_failed"

    # Still processing
    if upload.status in (
        UploadStatus.PENDING.value,
        UploadStatus.SCANNING.value,
        UploadStatus.EXTRACTING.value,
    ):
        return "processing"
    if upload.scan_status in ("pending", "scanning"):
        return "processing"
    if upload.extraction_status == "pending":
        return "processing"
    if upload.status == UploadStatus.FAILED.value:
        return "processing"

    # Deleted = rejected
    if upload.status == UploadStatus.DELETED.value:
        return "rejected"

    # Upload is completed and clean — check script linkage
    if upload.script_id is None:
        return "ready_for_conversion"

    # Script exists — check its lifecycle
    if script is None:
        return "ready_for_conversion"

    if script.is_deleted:
        return "rejected"
    if script.status == ScriptStatus.DRAFT.value:
        return "ready_for_review"
    if script.status == ScriptStatus.PUBLISHED.value:
        return "published"
    if script.status == ScriptStatus.UNPUBLISHED.value:
        return "rejected"

    return "processing"


# ─── Allowed-action calculation ───────────────────────────────────────────────


def derive_actions(review_status: str, upload: ScriptUpload) -> dict[str, bool]:
    """Deterministic server-side allowed-action calculation.

    Returns a dict with keys: can_edit, can_retry, can_reject, can_publish.
    """
    actions = {
        "can_edit": False,
        "can_retry": False,
        "can_reject": False,
        "can_publish": False,
    }

    if review_status == "ready_for_review":
        actions["can_edit"] = True
        actions["can_retry"] = True
        actions["can_reject"] = True
        actions["can_publish"] = True
    elif review_status == "ready_for_conversion":
        actions["can_retry"] = True  # retry conversion
        actions["can_reject"] = True
    elif review_status == "extraction_failed" or review_status == "scan_failed":
        actions["can_retry"] = _quarantine_source_exists(upload)
        actions["can_reject"] = True
    elif review_status == "infected":
        actions["can_reject"] = True

    return actions


def _quarantine_source_exists(upload: ScriptUpload) -> bool:
    """Check if the quarantined original file still exists on disk."""
    from app.services.upload_quarantine import get_quarantine_path

    try:
        quarantine_dir = get_quarantine_path()
        source_path = quarantine_dir / upload.storage_key
        return source_path.exists()
    except Exception:
        return False


# ─── Warnings ─────────────────────────────────────────────────────────────────


def build_warnings(
    upload: ScriptUpload,
    script: Script | None,
    review_status: str,
) -> list[ReviewWarning]:
    """Build structured warnings for the review detail.

    Never includes uploaded content inside warning messages.
    """
    warnings: list[ReviewWarning] = []

    # Sanitized extraction error (never raw exception message)
    if upload.extraction_error:
        # Only surface safe prefixed errors, never raw stack traces
        safe_msg = upload.extraction_error
        if len(safe_msg) > 200:
            safe_msg = safe_msg[:200] + "..."
        warnings.append(
            ReviewWarning(
                code="extraction_error",
                message=safe_msg,
                severity="error",
            )
        )

    # No scenario
    if upload.scenario_id is None:
        warnings.append(
            ReviewWarning(
                code="missing_scenario",
                message="Upload has no associated scenario. A scenario is required for conversion.",
                severity="error",
            )
        )

    # Content validation warnings (if script exists with draft)
    if script and script.draft_content:
        content_warnings = _validate_draft_content(script.draft_content)
        warnings.extend(content_warnings)

    # Quarantine expiry warning
    if upload.quarantine_expires_at:
        now = datetime.now(UTC)
        if upload.quarantine_expires_at < now and review_status not in ("published", "rejected"):
            warnings.append(
                ReviewWarning(
                    code="quarantine_expired",
                    message="Quarantine source file has expired. Retry of scan/extraction is unavailable.",
                    severity="warning",
                )
            )

    return warnings


def _validate_draft_content(draft_content: dict) -> list[ReviewWarning]:
    """Run validation on persisted draft content and return warnings if issues found."""
    warnings: list[ReviewWarning] = []

    try:
        raw_text = json.dumps(draft_content)
        limits = ScriptLimits(
            max_definition_size_bytes=settings.script_max_definition_size_bytes,
            max_trigger_phrases=settings.script_max_trigger_phrases,
            max_expected_replies=settings.script_max_expected_replies,
            max_escalation_conditions=settings.script_max_escalation_conditions,
            max_field_text_length=settings.script_max_field_text_length,
        )
        validate_script(raw_text, "json", limits)
    except ScriptFormatError:
        warnings.append(
            ReviewWarning(
                code="content_parse_error",
                message="Persisted script content cannot be parsed. Publication is blocked.",
                severity="error",
            )
        )
    except ScriptValidationError as exc:
        for error in exc.errors:
            field_path = ".".join(str(p) for p in error.get("loc", ()))
            warnings.append(
                ReviewWarning(
                    code="validation_error",
                    message=error.get("msg", "Unknown validation error"),
                    field=field_path or None,
                    severity="error",
                )
            )

    return warnings


# ─── Data loading ─────────────────────────────────────────────────────────────


async def load_upload_with_script(
    db: AsyncSession, upload_id: UUID
) -> tuple[ScriptUpload | None, Script | None]:
    """Load an upload and its linked script (if any)."""
    stmt = select(ScriptUpload).where(ScriptUpload.id == upload_id)
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()
    if upload is None:
        return None, None

    script = None
    if upload.script_id is not None:
        script_stmt = select(Script).where(
            Script.id == upload.script_id,
            Script.is_deleted == False,  # noqa: E712
        )
        script_result = await db.execute(script_stmt)
        script = script_result.scalar_one_or_none()

    return upload, script


# ─── Response building ────────────────────────────────────────────────────────


def build_review_detail(
    upload: ScriptUpload,
    script: Script | None,
) -> ReviewDetailResponse:
    """Build the full review detail response from persisted state.

    Security rules:
    - Infected uploads: sanitized_content and script_contract are omitted.
    - Never returns storage_key, quarantine paths, or scanner details.
    - Validates persisted content before returning; returns warning if invalid.
    """
    review_status = derive_review_status(upload, script)
    actions = derive_actions(review_status, upload)
    warnings = build_warnings(upload, script, review_status)

    # Build sanitized scan result (never expose connection details)
    scan_error_safe = None
    if upload.scan_status == "error":
        scan_error_safe = "Scanner was unavailable during processing"

    scan_result = ReviewScanResult(
        status=upload.scan_status,
        signature=upload.scan_signature if upload.scan_status == "infected" else None,
        error=scan_error_safe,
    )

    # Content: NEVER return for infected/unsafe uploads
    sanitized_content = None
    script_contract = None
    contract_format = None
    script_status = None

    is_safe = upload.scan_status not in ("infected",)

    if is_safe:
        sanitized_content = upload.extracted_content

    if script:
        script_status = script.status
        contract_format = script.format
        if is_safe and script.draft_content:
            try:
                from app.schemas.script import ScriptContract as ScriptContractSchema

                contract = ScriptContractSchema(**script.draft_content)
                script_contract = contract.model_dump(mode="json")
            except Exception:
                warnings.append(
                    ReviewWarning(
                        code="content_unparseable",
                        message="Persisted script content cannot be validated. Publication is blocked.",
                        severity="error",
                    )
                )
                script_contract = None

    return ReviewDetailResponse(
        upload_id=upload.id,
        filename_original=upload.filename_original,
        mime_type=upload.mime_type,
        file_size_bytes=upload.file_size_bytes,
        scenario_id=upload.scenario_id,
        uploaded_by=upload.uploaded_by,
        sanitized_content=sanitized_content,
        script_contract=script_contract,
        contract_format=contract_format,
        warnings=warnings,
        scan_result=scan_result,
        upload_status=upload.status,
        scan_status=upload.scan_status,
        extraction_status=upload.extraction_status,
        script_id=upload.script_id,
        script_status=script_status,
        review_status=review_status,
        can_edit=actions["can_edit"],
        can_retry=actions["can_retry"],
        can_reject=actions["can_reject"],
        can_publish=actions["can_publish"],
        rejection_reason=upload.rejection_reason if hasattr(upload, "rejection_reason") else None,
        rejected_by=upload.rejected_by if hasattr(upload, "rejected_by") else None,
        rejected_at=upload.rejected_at if hasattr(upload, "rejected_at") else None,
        created_at=upload.created_at,
        updated_at=upload.updated_at,
        published_at=None,  # Enriched by caller
    )


async def get_published_at(db: AsyncSession, script: Script) -> datetime | None:
    """Get the published_at timestamp for a published script's current version."""
    if script.status != ScriptStatus.PUBLISHED.value or script.current_version_id is None:
        return None
    stmt = select(ScriptVersion.published_at).where(ScriptVersion.id == script.current_version_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
