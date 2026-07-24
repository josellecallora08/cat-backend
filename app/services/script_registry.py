"""Script_Registry service: lifecycle orchestration and DB writes for Scripts.

Owns lifecycle transitions (draft -> published -> unpublished) and persistence.
Structural/format/limit validation itself lives in `app/services/script_validator.py`;
this module calls into that pure validation pipeline and then persists the result.

All mutating entry points take an already-authorized `admin_id`; authorization
is enforced at the API layer via `Depends(require_admin)` (see design.md), so
this module stays testable without needing to fake auth.
"""

import json
from typing import List, Optional
from uuid import UUID

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.script import Script, ScriptStatus, ScriptVersion
from app.services.script_validator import (
    ScriptLimits,
    parse_script_definition,
    validate_contract_structure,
    validate_script,
)


async def create_draft(
    db: AsyncSession,
    admin_id: UUID,
    name: str,
    scenario_id: UUID,
    format: str,
    raw_definition: str,
) -> Script:
    """Create a new Draft_Script (Requirements 3.1, 3.6).

    Parses `raw_definition` per `format` and structurally validates it against
    the Script_Contract shape (missing/invalid fields, Prohibited_Responses vs
    Expected_Replies conflicts) — NOT full limit validation, since a
    Draft_Script may be incomplete while still being authored. The parsed
    dict (not the validated model) is what gets persisted as `draft_content`,
    matching what an Administrator submitted.

    Args:
        db: Active async DB session.
        admin_id: ID of the Administrator creating the script.
        name: Human-readable name for the script.
        scenario_id: The scenario this script is tied to.
        format: Declared format of `raw_definition` ("json" or "yaml").
        raw_definition: The raw Script definition content.

    Returns:
        The newly persisted `Script` row, with `status="draft"`.

    Raises:
        ScriptFormatError: If `format` is unsupported or `raw_definition`
            cannot be parsed as the declared format.
        ScriptValidationError: If the parsed definition fails Script_Contract
            structural validation.
    """
    data = parse_script_definition(raw_definition, format)
    validate_contract_structure(data)

    script = Script(
        scenario_id=scenario_id,
        name=name,
        status=ScriptStatus.DRAFT.value,
        format=format,
        draft_content=data,
        created_by=admin_id,
    )
    db.add(script)
    await db.commit()
    await db.refresh(script)
    return script


async def get_script(db: AsyncSession, script_id: UUID) -> Optional[Script]:
    """Get a script by ID.

    Returns None if the script does not exist or has been soft-deleted.
    """
    stmt = select(Script).where(
        Script.id == script_id,
        Script.is_deleted == False,  # noqa: E712
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_scripts(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
) -> List[Script]:
    """List scripts with pagination, ordered by most recently created first.

    Excludes soft-deleted (is_deleted=True) scripts. Includes scripts of any
    status (draft, published, unpublished).
    """
    stmt = (
        select(Script)
        .where(Script.is_deleted == False)  # noqa: E712
        .order_by(Script.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_draft(
    db: AsyncSession,
    script_id: UUID,
    raw_definition: str,
    format: str,
) -> Script:
    """Replace the draft content of an existing Script (Requirements 3.2, 3.9).

    Parses `raw_definition` per `format` and structurally validates it against
    the Script_Contract shape (contract-shape only, not full limit
    validation), matching `create_draft`'s validation strictness.

    If the target `Script` is currently `draft`, this replaces its draft
    content outright. If the target `Script` is currently `published`, per
    Requirement 3.9 this instead layers a new draft revision on top: it
    overwrites `draft_content`/`format` on the same `Script` row without
    touching `status`, `current_version_id`, or any `ScriptVersion` row, so
    the existing published version remains completely untouched and the
    script stays consumable. Regardless of current status, `Script.status`
    itself is never modified by this function.

    Args:
        db: Active async DB session.
        script_id: ID of the Script to update.
        raw_definition: The raw Script definition content.
        format: Declared format of `raw_definition` ("json" or "yaml").

    Returns:
        The updated `Script` row.

    Raises:
        ValueError: If no Script with `script_id` exists (or it is
            soft-deleted).
        ScriptFormatError: If `format` is unsupported or `raw_definition`
            cannot be parsed as the declared format.
        ScriptValidationError: If the parsed definition fails Script_Contract
            structural validation.
    """
    script = await get_script(db, script_id)
    if script is None:
        raise ValueError(f"Script with id {script_id} not found")

    data = parse_script_definition(raw_definition, format)
    validate_contract_structure(data)

    script.draft_content = data
    script.format = format
    await db.commit()
    await db.refresh(script)
    return script


async def publish(db: AsyncSession, script_id: UUID, admin_id: UUID) -> ScriptVersion:
    """Publish a Script's current draft content (Requirements 3.7, 3.8).

    Runs the draft content through the full `validate_script` pipeline
    (contract structure + conflicts + configurable limits) exactly as a
    fresh submission would be validated. On success, creates a new
    immutable `ScriptVersion` (`version_number` = previous max + 1 for
    this script, or 1 if none exist) and updates the `Script` row's
    `status` to `"published"` and `current_version_id` to point at the
    new version, all within a single transaction.

    On any validation failure, the `Script` row is left completely
    unchanged (Requirement 3.8) — the exception is propagated before any
    write is attempted.

    Args:
        db: Active async DB session.
        script_id: ID of the Script to publish.
        admin_id: ID of the Administrator publishing the script; recorded
            as the new `ScriptVersion.published_by`.

    Returns:
        The newly created `ScriptVersion`.

    Raises:
        ValueError: If no Script with `script_id` exists (or it is
            soft-deleted).
        ScriptFormatError: If the Script's `draft_content` cannot be
            re-serialized/parsed as its declared `format` (should not
            normally occur, since `draft_content` is always persisted
            already-parsed).
        ScriptValidationError: If the draft content fails Script_Contract
            structural validation, conflict checks, or configurable
            limits. The `Script` row is left completely unchanged.
    """
    script = await get_script(db, script_id)
    if script is None:
        raise ValueError(f"Script with id {script_id} not found")

    if script.format == "json":
        raw_text = json.dumps(script.draft_content)
    else:
        raw_text = yaml.safe_dump(script.draft_content)

    limits = ScriptLimits(
        max_definition_size_bytes=settings.script_max_definition_size_bytes,
        max_trigger_phrases=settings.script_max_trigger_phrases,
        max_expected_replies=settings.script_max_expected_replies,
        max_escalation_conditions=settings.script_max_escalation_conditions,
        max_field_text_length=settings.script_max_field_text_length,
    )

    # Raises ScriptFormatError/ScriptValidationError on failure. No write
    # has been attempted yet, so the Script row is left untouched.
    validated_contract = validate_script(raw_text, script.format, limits)

    max_version_stmt = select(func.max(ScriptVersion.version_number)).where(
        ScriptVersion.script_id == script_id
    )
    result = await db.execute(max_version_stmt)
    current_max_version = result.scalar()

    new_version = ScriptVersion(
        script_id=script_id,
        version_number=(current_max_version or 0) + 1,
        content=validated_contract.model_dump(mode="json"),
        published_by=admin_id,
    )
    db.add(new_version)
    await db.flush()

    script.status = ScriptStatus.PUBLISHED.value
    script.current_version_id = new_version.id

    await db.commit()
    await db.refresh(new_version)
    return new_version


async def unpublish(db: AsyncSession, script_id: UUID, admin_id: UUID) -> Script:
    """Unpublish a Published_Script (Requirement 3.10).

    Sets `Script.status` to `"unpublished"` while leaving
    `current_version_id` and every existing `ScriptVersion` row completely
    untouched, so a previously published version remains intact (just no
    longer active for new consumption per
    `get_active_published_version`/Training_Call creation).

    Args:
        db: Active async DB session.
        script_id: ID of the Script to unpublish.
        admin_id: ID of the Administrator unpublishing the script (accepted
            for API-layer symmetry/future audit logging; authorization
            itself is enforced at the API layer).

    Returns:
        The updated `Script` row.

    Raises:
        ValueError: If no Script with `script_id` exists (or it is
            soft-deleted).
    """
    script = await get_script(db, script_id)
    if script is None:
        raise ValueError(f"Script with id {script_id} not found")

    script.status = ScriptStatus.UNPUBLISHED.value
    await db.commit()
    await db.refresh(script)
    return script


async def delete_script(db: AsyncSession, script_id: UUID, admin_id: UUID) -> None:
    """Soft-delete a Script (Requirement 3.4).

    Sets `Script.is_deleted` to `True` without deleting or modifying any
    `ScriptVersion` row referencing this script, and without touching
    `status`/`current_version_id`. Soft-deleted scripts are excluded from
    `get_script`/`list_scripts` (both already filter on
    `is_deleted == False`).

    Args:
        db: Active async DB session.
        script_id: ID of the Script to soft-delete.
        admin_id: ID of the Administrator deleting the script (accepted
            for API-layer symmetry/future audit logging; authorization
            itself is enforced at the API layer).

    Returns:
        None.

    Raises:
        ValueError: If no Script with `script_id` exists (or it is already
            soft-deleted).
    """
    script = await get_script(db, script_id)
    if script is None:
        raise ValueError(f"Script with id {script_id} not found")

    script.is_deleted = True
    await db.commit()
    return None


async def get_active_published_version(
    db: AsyncSession, scenario_id: UUID
) -> Optional[ScriptVersion]:
    """Get the active Published_Script's current version for a scenario
    (Requirements 4.1, 4.3).

    Looks up the (at most one, since `scenario_id` is unique on `Script`)
    non-soft-deleted `Script` row for `scenario_id`, and returns its
    `current_version_id`'s `ScriptVersion` only when the script's
    `status` is exactly `"published"`. Draft-only and unpublished scripts
    are treated identically to having no script at all, since neither is
    a consumable Published_Script.

    Args:
        db: Active async DB session.
        scenario_id: ID of the scenario to look up the active
            Published_Script for.

    Returns:
        The `ScriptVersion` row referenced by the scenario's `Script`'s
        `current_version_id`, if and only if that `Script` exists
        (non-soft-deleted) and its `status == "published"`. Otherwise
        `None` — including when no `Script` exists for `scenario_id`,
        when the `Script` is still a Draft_Script, when it has been
        unpublished, or (defensively, should not normally occur given
        `publish()`'s invariants) when `status == "published"` but
        `current_version_id` is `None`.
    """
    stmt = select(Script).where(
        Script.scenario_id == scenario_id,
        Script.is_deleted == False,  # noqa: E712
    )
    result = await db.execute(stmt)
    script = result.scalar_one_or_none()

    if script is None:
        return None

    if script.status != ScriptStatus.PUBLISHED.value:
        return None

    if script.current_version_id is None:
        return None

    version_stmt = select(ScriptVersion).where(
        ScriptVersion.id == script.current_version_id
    )
    version_result = await db.execute(version_stmt)
    return version_result.scalar_one_or_none()
