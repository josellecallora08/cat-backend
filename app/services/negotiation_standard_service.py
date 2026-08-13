"""Lifecycle services for administrator-managed negotiation standards."""

import copy
import hashlib
import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Campaign, NegotiationStandard, NegotiationStandardVersion, Session
from app.schemas.negotiation_standard import (
    NegotiationStandardContent,
    ValidationResult,
)
from app.services.negotiation_standard_validator import validate_standard


logger = logging.getLogger(__name__)


class StandardConflictError(RuntimeError):
    """Raised when a standard cannot be changed because of its lifecycle/state."""


class StandardNotFoundError(LookupError):
    """Raised when a campaign standard or version does not exist."""


class StandardValidationError(ValueError):
    """Raised when publication validation fails."""

    def __init__(self, result: ValidationResult) -> None:
        super().__init__("Negotiation standard failed validation")
        self.result = result


def canonical_content_hash(content: NegotiationStandardContent | dict) -> str:
    """Return the SHA-256 hash of a compact, sorted canonical snapshot."""
    payload = (
        content.model_dump(mode="json")
        if isinstance(content, NegotiationStandardContent)
        else content
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(content: NegotiationStandardContent) -> dict:
    """Deep-copy serialized rubric content for immutable persistence."""
    return copy.deepcopy(content.model_dump(mode="json"))


def _audit(event_name: str, standard: NegotiationStandard, admin_id: UUID, **extra: object) -> None:
    """Emit an audit event without rubric, transcript, prompt, or response data."""
    fields = {
        "event": event_name,
        "standard_id": str(standard.id),
        "campaign_id": str(standard.campaign_id),
        "admin_id": str(admin_id),
        "status": standard.status,
        "revision": standard.revision,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    fields.update({key: str(value) for key, value in extra.items()})
    logger.info(event_name, extra=fields)


async def _get_standard(
    db: AsyncSession, campaign_id: UUID, *, lock: bool = False
) -> NegotiationStandard:
    """Load a campaign standard or raise a typed not-found error."""
    statement = select(NegotiationStandard).where(NegotiationStandard.campaign_id == campaign_id)
    if lock:
        statement = statement.with_for_update()
    standard = (await db.execute(statement)).scalar_one_or_none()
    if standard is None:
        raise StandardNotFoundError("Negotiation standard not found")
    return standard


async def create_standard(
    db: AsyncSession,
    campaign_id: UUID,
    admin_id: UUID,
    name: str,
    description: str | None,
    content: NegotiationStandardContent,
) -> NegotiationStandard:
    """Create a draft standard and preserve incomplete publication weights."""
    campaign = (
        await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    ).scalar_one_or_none()
    if campaign is None:
        raise StandardNotFoundError("Campaign not found")
    existing = (
        await db.execute(
            select(NegotiationStandard).where(NegotiationStandard.campaign_id == campaign_id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise StandardConflictError("Campaign already has a negotiation standard")

    standard = NegotiationStandard(
        campaign_id=campaign_id,
        name=name,
        description=description,
        overall_passing_score=content.overall_passing_score,
        draft_content=_snapshot(content),
        created_by=admin_id,
        updated_by=admin_id,
    )
    db.add(standard)
    await db.commit()
    await db.refresh(standard)
    _audit("negotiation_standard_created", standard, admin_id)
    return standard


async def get_standard(db: AsyncSession, campaign_id: UUID) -> NegotiationStandard:
    """Return the current campaign standard."""
    return await _get_standard(db, campaign_id)


async def update_draft(
    db: AsyncSession,
    campaign_id: UUID,
    admin_id: UUID,
    *,
    expected_revision: int,
    name: str | None = None,
    description: str | None = None,
    content: NegotiationStandardContent | None = None,
) -> NegotiationStandard:
    """Update a draft using optimistic revision concurrency."""
    standard = await _get_standard(db, campaign_id, lock=True)
    if standard.status != "draft":
        raise StandardConflictError("Only draft negotiation standards can be updated")
    if standard.revision != expected_revision:
        raise StandardConflictError("Negotiation standard revision is stale")
    if name is not None:
        standard.name = name
    if description is not None:
        standard.description = description
    if content is not None:
        standard.overall_passing_score = content.overall_passing_score
        standard.draft_content = _snapshot(content)
    standard.updated_by = admin_id
    standard.revision += 1
    await db.commit()
    await db.refresh(standard)
    _audit("negotiation_standard_updated", standard, admin_id)
    return standard


async def delete_draft(db: AsyncSession, campaign_id: UUID, admin_id: UUID) -> None:
    """Delete an unreferenced draft standard."""
    standard = await _get_standard(db, campaign_id, lock=True)
    if standard.status != "draft":
        raise StandardConflictError("Only draft negotiation standards can be deleted")
    version_exists = (
        await db.execute(
            select(NegotiationStandardVersion.id)
            .where(NegotiationStandardVersion.standard_id == standard.id)
            .limit(1)
        )
    ).scalar_one_or_none()
    pinned_exists = (
        await db.execute(
            select(Session.id)
            .where(
                Session.negotiation_standard_version_id.in_(
                    select(NegotiationStandardVersion.id).where(
                        NegotiationStandardVersion.standard_id == standard.id
                    )
                )
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if version_exists is not None or pinned_exists is not None:
        raise StandardConflictError(
            "Negotiation standard has published versions or pinned sessions"
        )
    await db.delete(standard)
    await db.commit()
    _audit("negotiation_standard_deleted", standard, admin_id)


async def validate_draft(db: AsyncSession, campaign_id: UUID, admin_id: UUID) -> ValidationResult:
    """Validate a draft and emit a safe validation audit event."""
    standard = await _get_standard(db, campaign_id)
    content = NegotiationStandardContent.model_validate(standard.draft_content or {})
    result = validate_standard(content, for_publish=False)
    _audit("negotiation_standard_validated", standard, admin_id, valid=result.valid)
    return result


async def publish_standard(
    db: AsyncSession,
    campaign_id: UUID,
    admin_id: UUID,
    publication_note: str | None = None,
) -> NegotiationStandardVersion:
    """Validate and publish a new immutable version, idempotently by content hash."""
    standard = await _get_standard(db, campaign_id, lock=True)
    if standard.status == "archived":
        raise StandardConflictError("Archived negotiation standards cannot be published")
    content = NegotiationStandardContent.model_validate(standard.draft_content or {})
    validation = validate_standard(content, for_publish=True)
    if not validation.valid:
        raise StandardValidationError(validation)

    snapshot = _snapshot(content)
    content_hash = canonical_content_hash(snapshot)
    current = standard.current_version
    if current is None:
        current = (
            await db.execute(
                select(NegotiationStandardVersion)
                .where(NegotiationStandardVersion.standard_id == standard.id)
                .order_by(NegotiationStandardVersion.version_number.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    if current is not None and current.content_hash == content_hash:
        return current

    next_number = (current.version_number + 1) if current is not None else 1
    version = NegotiationStandardVersion(
        standard_id=standard.id,
        version_number=next_number,
        schema_version=content.schema_version,
        snapshot=snapshot,
        content_hash=content_hash,
        created_by=admin_id,
        published_by=admin_id,
        publication_note=publication_note,
    )
    db.add(version)
    await db.flush()
    standard.current_version_id = version.id
    standard.status = "published"
    standard.updated_by = admin_id
    standard.revision += 1
    await db.commit()
    await db.refresh(version)
    _audit(
        "negotiation_standard_published",
        standard,
        admin_id,
        version_id=version.id,
        version_number=version.version_number,
    )
    return version


async def archive_standard(
    db: AsyncSession, campaign_id: UUID, admin_id: UUID
) -> NegotiationStandard:
    """Archive a standard; repeated archive requests are safe and idempotent."""
    standard = await _get_standard(db, campaign_id, lock=True)
    if standard.status == "archived":
        return standard
    standard.status = "archived"
    standard.updated_by = admin_id
    standard.revision += 1
    await db.commit()
    await db.refresh(standard)
    _audit("negotiation_standard_archived", standard, admin_id)
    return standard


async def list_versions(
    db: AsyncSession,
    campaign_id: UUID,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[NegotiationStandardVersion], int]:
    """Return newest-first versions with a total count."""
    standard = await _get_standard(db, campaign_id)
    total = await db.scalar(
        select(func.count())
        .select_from(NegotiationStandardVersion)
        .where(NegotiationStandardVersion.standard_id == standard.id)
    )
    versions = (
        (
            await db.execute(
                select(NegotiationStandardVersion)
                .where(NegotiationStandardVersion.standard_id == standard.id)
                .order_by(NegotiationStandardVersion.version_number.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return list(versions), int(total or 0)


async def get_version(
    db: AsyncSession, campaign_id: UUID, version_id: UUID
) -> NegotiationStandardVersion:
    """Return a version belonging to the requested campaign."""
    standard = await _get_standard(db, campaign_id)
    version = (
        await db.execute(
            select(NegotiationStandardVersion).where(
                NegotiationStandardVersion.id == version_id,
                NegotiationStandardVersion.standard_id == standard.id,
            )
        )
    ).scalar_one_or_none()
    if version is None:
        raise StandardNotFoundError("Negotiation standard version not found")
    return version
