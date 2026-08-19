"""Administrator API for campaign negotiation standards."""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import NegotiationStandard, NegotiationStandardVersion, User
from app.schemas.negotiation_standard import NegotiationStandardContent, ValidationIssue
from app.services.auth import require_admin
from app.services.negotiation_standard_service import (
    StandardConflictError,
    StandardNotFoundError,
    StandardValidationError,
    archive_standard,
    create_standard,
    delete_draft,
    get_standard,
    get_version,
    list_versions,
    publish_standard,
    reopen_draft,
    update_draft,
    validate_draft,
)


router = APIRouter()


class CreateStandardRequest(BaseModel):
    """Payload for creating a campaign standard draft."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    draft_content: NegotiationStandardContent


class UpdateStandardRequest(BaseModel):
    """Payload for updating a draft with optimistic concurrency."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    draft_content: NegotiationStandardContent | None = None


class PublishRequest(BaseModel):
    """Optional note attached to a published version."""

    model_config = ConfigDict(extra="forbid")

    publication_note: str | None = Field(default=None, max_length=500)


class StandardResponse(BaseModel):
    """Current standard and its draft/current-version metadata."""

    id: UUID
    campaign_id: UUID
    name: str
    description: str | None
    status: str
    revision: int
    draft_content: dict | None
    current_version_id: UUID | None
    current_version_number: int | None


class VersionResponse(BaseModel):
    """Immutable published version response."""

    id: UUID
    standard_id: UUID
    version_number: int
    schema_version: int
    snapshot: dict
    content_hash: str
    created_by: UUID
    published_by: UUID
    created_at: datetime
    published_at: datetime
    publication_note: str | None


class VersionPage(BaseModel):
    """Paginated version history."""

    items: list[VersionResponse]
    page: int
    page_size: int
    total: int


class ValidationResponse(BaseModel):
    """Structured aggregate validation response."""

    valid: bool
    weight_total: int
    errors: list[ValidationIssue]


def _standard_response(standard: NegotiationStandard) -> StandardResponse:
    """Map an ORM standard to its API response."""
    current = standard.current_version
    return StandardResponse(
        id=standard.id,
        campaign_id=standard.campaign_id,
        name=standard.name,
        description=standard.description,
        status=standard.status,
        revision=standard.revision,
        draft_content=standard.draft_content,
        current_version_id=current.id if current else None,
        current_version_number=current.version_number if current else None,
    )


def _version_response(version: NegotiationStandardVersion) -> VersionResponse:
    """Map an immutable ORM version to its API response."""
    return VersionResponse.model_validate(version, from_attributes=True)


def _raise_service_error(error: Exception) -> None:
    """Translate service domain errors into stable HTTP contracts."""
    if isinstance(error, StandardNotFoundError):
        raise HTTPException(
            status_code=404, detail={"code": "not_found", "message": str(error)}
        ) from error
    if isinstance(error, StandardConflictError):
        raise HTTPException(
            status_code=409, detail={"code": "conflict", "message": str(error)}
        ) from error
    if isinstance(error, StandardValidationError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_failed",
                "weight_total": error.result.weight_total,
                "errors": [issue.model_dump() for issue in error.result.errors],
            },
        ) from error
    raise error


@router.post("", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def create_standard_endpoint(
    campaign_id: UUID,
    body: CreateStandardRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> StandardResponse:
    """Create a draft standard for a campaign."""
    try:
        standard = await create_standard(
            db, campaign_id, admin.id, body.name, body.description, body.draft_content
        )
    except (StandardNotFoundError, StandardConflictError) as error:
        _raise_service_error(error)
    return _standard_response(standard)


@router.get("", response_model=StandardResponse)
async def get_standard_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> StandardResponse:
    """Read the current campaign standard."""
    try:
        return _standard_response(await get_standard(db, campaign_id))
    except StandardNotFoundError as error:
        _raise_service_error(error)
        raise AssertionError("unreachable") from error


@router.put("", response_model=StandardResponse)
async def update_standard_endpoint(
    campaign_id: UUID,
    body: UpdateStandardRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> StandardResponse:
    """Update a draft standard."""
    try:
        standard = await update_draft(
            db,
            campaign_id,
            admin.id,
            expected_revision=body.expected_revision,
            name=body.name,
            description=body.description,
            content=body.draft_content,
        )
    except (StandardNotFoundError, StandardConflictError) as error:
        _raise_service_error(error)
    return _standard_response(standard)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_standard_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> Response:
    """Delete an unreferenced draft standard."""
    try:
        await delete_draft(db, campaign_id, admin.id)
    except (StandardNotFoundError, StandardConflictError) as error:
        _raise_service_error(error)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/validate", response_model=ValidationResponse)
async def validate_standard_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ValidationResponse:
    """Validate the current draft without publishing it."""
    try:
        result = await validate_draft(db, campaign_id, admin.id)
    except StandardNotFoundError as error:
        _raise_service_error(error)
    return ValidationResponse.model_validate(result.model_dump())


@router.post("/publish", response_model=VersionResponse, status_code=status.HTTP_201_CREATED)
async def publish_standard_endpoint(
    campaign_id: UUID,
    body: PublishRequest | None = None,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> VersionResponse:
    """Validate and publish a new immutable version."""
    try:
        version = await publish_standard(
            db, campaign_id, admin.id, body.publication_note if body else None
        )
    except (StandardNotFoundError, StandardConflictError, StandardValidationError) as error:
        _raise_service_error(error)
    return _version_response(version)


@router.post("/archive", response_model=StandardResponse)
async def archive_standard_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> StandardResponse:
    """Archive a campaign standard."""
    try:
        standard = await archive_standard(db, campaign_id, admin.id)
    except (StandardNotFoundError, StandardConflictError) as error:
        _raise_service_error(error)
    return _standard_response(standard)


@router.post("/reopen", response_model=StandardResponse)
async def reopen_standard_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> StandardResponse:
    """Reopen a published or archived standard as an editable draft.

    Existing published versions and any sessions pinned to them are
    unaffected; only the mutable standard's status flips back to `draft` so
    its content can be edited and republished as a new version.
    """
    try:
        standard = await reopen_draft(db, campaign_id, admin.id)
    except (StandardNotFoundError, StandardConflictError) as error:
        _raise_service_error(error)
    return _standard_response(standard)


@router.get("/versions", response_model=VersionPage)
async def list_versions_endpoint(
    campaign_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> VersionPage:
    """List immutable versions newest-first."""
    try:
        versions, total = await list_versions(db, campaign_id, page, page_size)
    except StandardNotFoundError as error:
        _raise_service_error(error)
    return VersionPage(
        items=[_version_response(version) for version in versions],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/versions/{version_id}", response_model=VersionResponse)
async def get_version_endpoint(
    campaign_id: UUID,
    version_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> VersionResponse:
    """Read one immutable version belonging to the campaign."""
    try:
        return _version_response(await get_version(db, campaign_id, version_id))
    except StandardNotFoundError as error:
        _raise_service_error(error)
        raise AssertionError("unreachable") from error
