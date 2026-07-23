"""Script Registry management API endpoints (admin only)."""

import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.script import Script, ScriptVersion
from app.models.user import User
from app.schemas.script import (
    PaginatedScripts,
    ScriptCreateRequest,
    ScriptDetail,
    ScriptListItem,
    ScriptUpdateRequest,
    ScriptVersionItem,
)
from app.services.auth import require_admin
from app.services.script_registry import (
    create_draft,
    delete_script,
    get_script,
    list_scripts,
    publish,
    unpublish,
    update_draft,
)
from app.services.script_validator import (
    ScriptFormatError,
    ScriptLimitError,
    ScriptValidationError,
)


router = APIRouter()


def _script_to_detail(script: Script) -> ScriptDetail:
    """Convert a Script ORM instance to a ScriptDetail response schema."""
    return ScriptDetail(
        id=script.id,
        name=script.name,
        scenario_id=script.scenario_id,
        status=script.status,
        format=script.format,
        draft_content=script.draft_content,
        current_version_id=script.current_version_id,
        created_at=script.created_at,
        updated_at=script.updated_at,
    )


def _script_to_list_item(script: Script) -> ScriptListItem:
    """Convert a Script ORM instance to a ScriptListItem response schema."""
    return ScriptListItem(
        id=script.id,
        name=script.name,
        scenario_id=script.scenario_id,
        status=script.status,
        format=script.format,
        created_at=script.created_at,
        updated_at=script.updated_at,
    )


def _version_to_item(version: ScriptVersion) -> ScriptVersionItem:
    """Convert a ScriptVersion ORM instance to a ScriptVersionItem response schema."""
    return ScriptVersionItem(
        id=version.id,
        version_number=version.version_number,
        content=version.content,
        published_by=version.published_by,
        published_at=version.published_at,
    )


def _validation_error_detail(exc: Exception) -> dict:
    """Build an HTTPException detail payload surfacing every validation
    error/limit violation carried by a script_validator exception, so an
    admin can see exactly what's wrong rather than just the first issue.
    """
    detail: dict = {"message": str(exc)}
    errors = getattr(exc, "errors", None)
    if errors is not None:
        detail["errors"] = errors
    violations = getattr(exc, "violations", None)
    if violations is not None:
        detail["violations"] = violations
    return detail


@router.post("", response_model=ScriptDetail, status_code=201)
async def create_script_endpoint(
    body: ScriptCreateRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ScriptDetail:
    """Create a new Draft_Script (Requirements 3.1, 3.6)."""
    try:
        script = await create_draft(
            db,
            admin_id=admin.id,
            name=body.name,
            scenario_id=body.scenario_id,
            format=body.format,
            raw_definition=body.raw_definition,
        )
    except (ScriptFormatError, ScriptValidationError) as e:
        raise HTTPException(status_code=422, detail=_validation_error_detail(e))
    return _script_to_detail(script)


@router.get("", response_model=PaginatedScripts)
async def list_scripts_endpoint(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> PaginatedScripts:
    """List scripts with pagination (Requirement 3.3)."""
    count_stmt = select(func.count(Script.id)).where(
        Script.is_deleted == False  # noqa: E712
    )
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0
    total_pages = math.ceil(total / page_size) if total > 0 else 0

    scripts = await list_scripts(db, page, page_size)

    return PaginatedScripts(
        items=[_script_to_list_item(s) for s in scripts],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{script_id}", response_model=ScriptDetail)
async def get_script_endpoint(
    script_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ScriptDetail:
    """Get full script detail (Requirement 3.3)."""
    script = await get_script(db, script_id)
    if script is None:
        raise HTTPException(status_code=404, detail="Script not found")
    return _script_to_detail(script)


@router.put("/{script_id}", response_model=ScriptDetail)
async def update_script_endpoint(
    script_id: UUID,
    body: ScriptUpdateRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ScriptDetail:
    """Replace a script's draft content (Requirements 3.2, 3.9)."""
    try:
        script = await update_draft(
            db,
            script_id=script_id,
            raw_definition=body.raw_definition,
            format=body.format,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ScriptFormatError, ScriptValidationError) as e:
        raise HTTPException(status_code=422, detail=_validation_error_detail(e))
    return _script_to_detail(script)


@router.delete("/{script_id}", status_code=204)
async def delete_script_endpoint(
    script_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> None:
    """Soft-delete a script (Requirement 3.4)."""
    try:
        await delete_script(db, script_id, admin_id=admin.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{script_id}/publish", response_model=ScriptDetail)
async def publish_script_endpoint(
    script_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ScriptDetail:
    """Publish a script's current draft content (Requirements 3.7, 3.8)."""
    try:
        await publish(db, script_id, admin_id=admin.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ScriptFormatError, ScriptValidationError, ScriptLimitError) as e:
        raise HTTPException(status_code=422, detail=_validation_error_detail(e))

    script = await get_script(db, script_id)
    if script is None:
        raise HTTPException(status_code=404, detail="Script not found")
    return _script_to_detail(script)


@router.post("/{script_id}/unpublish", response_model=ScriptDetail)
async def unpublish_script_endpoint(
    script_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ScriptDetail:
    """Unpublish a published script (Requirement 3.10)."""
    try:
        script = await unpublish(db, script_id, admin_id=admin.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _script_to_detail(script)


@router.get("/{script_id}/versions", response_model=list[ScriptVersionItem])
async def list_script_versions_endpoint(
    script_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> list[ScriptVersionItem]:
    """List all published versions of a script, oldest first."""
    script = await get_script(db, script_id)
    if script is None:
        raise HTTPException(status_code=404, detail="Script not found")

    stmt = (
        select(ScriptVersion)
        .where(ScriptVersion.script_id == script_id)
        .order_by(ScriptVersion.version_number)
    )
    result = await db.execute(stmt)
    versions = list(result.scalars().all())
    return [_version_to_item(v) for v in versions]
