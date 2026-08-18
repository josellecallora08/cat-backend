"""Parse and validate configurable local JSON rubric seed sources."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.config import settings
from app.database import async_session_factory
from app.models import (
    Campaign,
    CampaignAgent,
    CampaignRole,
    CampaignStatus,
    NegotiationStandard,
    NegotiationStandardVersion,
    Scenario,
    Session,
)
from app.models.campaign import campaign_scenarios
from app.models.user import User, UserRole, UserType
from app.schemas.negotiation_standard import NegotiationStandardContent
from app.services.audit import log_rubric_seed_completed
from app.services.negotiation_standard_service import (
    StandardConflictError,
    StandardValidationError,
    canonical_content_hash,
)
from app.services.negotiation_standard_validator import validate_standard


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


SourceErrorCode = Literal["configuration", "unavailable", "malformed", "unsupported"]


class RubricSourceError(ValueError):
    """Safe, classified error raised while reading or validating a rubric source."""

    def __init__(self, code: SourceErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


SeedStatus = Literal["success", "partial_success", "failed", "disabled"]
DefinitionStatus = Literal["accepted", "reused", "created", "rejected", "failed"]
SeedErrorCode = Literal[
    "configuration",
    "unavailable",
    "malformed",
    "unsupported",
    "persistence",
    "concurrency",
    "validation",
]


class ClassifiedSeedError(BaseModel):
    """Safe, machine-readable error detail for a seed run or definition."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: SeedErrorCode
    message: str = Field(min_length=1, max_length=300)


class DefinitionOutcome(BaseModel):
    """Safe outcome for one rubric definition, without source or infrastructure details."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    identity: str = Field(min_length=1, max_length=120)
    status: DefinitionStatus
    version_id: UUID | None = None
    error: ClassifiedSeedError | None = None
    published: bool = False


class SeedRunResult(BaseModel):
    """Structured, secret-free summary of one rubric seed execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    source_id: str = Field(min_length=1, max_length=120)
    status: SeedStatus
    reused_rubrics: int = 0
    created_rubrics: int = 0
    reused_versions: int = 0
    created_versions: int = 0
    published_versions: int = 0
    rejected_definitions: int = 0
    warnings: list[str] = Field(default_factory=list, max_length=100)
    definition_outcomes: list[DefinitionOutcome] = Field(default_factory=list, max_length=1000)
    campaign_id: UUID | None = None
    scenario_id: UUID | None = None
    agent_id: UUID | None = None
    rubric_version_id: UUID | None = None
    session_id: UUID | None = None

    @classmethod
    def new(cls, source_id: str, *, disabled: bool = False) -> SeedRunResult:
        """Create an empty result with a safe source identifier and initial status."""
        safe_source_id = _safe_source_identifier(source_id)
        return cls(
            run_id=uuid4(),
            source_id=safe_source_id,
            status="disabled" if disabled else "success",
        )

    def add_outcome(
        self,
        outcome: DefinitionOutcome,
        *,
        rubric_created: bool = False,
        version_created: bool = False,
    ) -> None:
        """Append an outcome and update counters without exposing unsafe details."""
        self.definition_outcomes.append(outcome)
        is_accepted = outcome.status in {"accepted", "reused", "created"}
        self.created_rubrics += int(is_accepted and rubric_created)
        self.reused_rubrics += int(is_accepted and not rubric_created)
        self.created_versions += int(is_accepted and version_created)
        self.reused_versions += int(is_accepted and not version_created)
        self.published_versions += int(outcome.published)
        self.rejected_definitions += int(outcome.status == "rejected")
        if outcome.status == "failed":
            self.status = "failed"
        elif outcome.status == "rejected" and self._has_success():
            self.status = "partial_success"
        elif outcome.status == "rejected":
            self.status = "failed"

    def _has_success(self) -> bool:
        return any(
            item.status in {"accepted", "reused", "created"} for item in self.definition_outcomes
        )


class RubricDefinition(BaseModel):
    """One strict rubric definition from a seed source."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    campaign_id: UUID
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    draft_content: NegotiationStandardContent
    publish: bool = False
    publication_note: str | None = Field(default=None, max_length=1000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject names that become empty after normalization."""
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("name must contain non-whitespace characters")
        return normalized


class SeedStandardConflictError(RuntimeError):
    """Raised when a campaign or source identity already owns another standard."""


async def get_or_create_seed_standard(
    db: AsyncSession,
    definition: RubricDefinition,
    *,
    source_id: str,
    admin_id: UUID,
) -> tuple[NegotiationStandard, bool]:
    """Lock ownership and return the standard for one source definition.

    The helper only flushes new rows; transaction boundaries remain with the caller so
    later version and publication work can commit or roll back the whole definition.
    """
    normalized_source = _safe_source_identifier(source_id)
    identity_key = normalize_rubric_identity(definition.name)
    campaign = (
        await db.execute(
            select(Campaign).where(Campaign.id == definition.campaign_id).with_for_update()
        )
    ).scalar_one_or_none()
    if campaign is None:
        raise SeedStandardConflictError("Campaign for rubric definition was not found.")

    standard = (
        await db.execute(
            select(NegotiationStandard)
            .where(
                NegotiationStandard.source_id == normalized_source,
                NegotiationStandard.source_rubric_key == identity_key,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if standard is not None:
        if standard.campaign_id != definition.campaign_id:
            raise SeedStandardConflictError("Rubric identity belongs to another campaign.")
        return standard, False

    existing_campaign_standard = (
        await db.execute(
            select(NegotiationStandard)
            .where(NegotiationStandard.campaign_id == definition.campaign_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if existing_campaign_standard is not None:
        if (
            existing_campaign_standard.source_id != normalized_source
            or existing_campaign_standard.source_rubric_key != identity_key
        ):
            raise SeedStandardConflictError("Campaign already has a different rubric standard.")
        return existing_campaign_standard, False

    content = normalize_rubric_content(definition.draft_content)
    standard = NegotiationStandard(
        campaign_id=definition.campaign_id,
        name=definition.name,
        source_id=normalized_source,
        source_rubric_key=identity_key,
        description=definition.description,
        overall_passing_score=content.overall_passing_score,
        draft_content=canonical_rubric_content(content),
        created_by=admin_id,
        updated_by=admin_id,
    )
    db.add(standard)
    await db.flush()
    return standard, True


def normalize_rubric_identity(name: str) -> str:
    """Normalize a rubric name for stable source identity matching."""
    return " ".join(name.split()).casefold()


class RubricSourceDocument(BaseModel):
    """Strict top-level document accepted by the rubric seed flow."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=120)
    definitions: list[RubricDefinition]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        """Normalize the source identifier while retaining a safe bounded value."""
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("source_id must contain non-whitespace characters")
        return normalized


def normalize_rubric_content(
    content: NegotiationStandardContent | dict[str, Any],
) -> NegotiationStandardContent:
    """Validate and normalize rubric content through the canonical Pydantic schema.

    A fresh model is returned even when ``content`` is already a model, preventing
    callers from carrying mutable input dictionaries into persistence.
    """
    parsed = (
        content
        if isinstance(content, NegotiationStandardContent)
        else NegotiationStandardContent.model_validate(content)
    )
    return NegotiationStandardContent.model_validate(parsed.model_dump(mode="json"))


def canonical_rubric_content(
    content: NegotiationStandardContent | dict[str, Any],
) -> dict[str, Any]:
    """Return the normalized JSON-compatible rubric snapshot used for persistence."""
    return normalize_rubric_content(content).model_dump(mode="json")


def rubric_content_hash(content: NegotiationStandardContent | dict[str, Any]) -> str:
    """Return the existing service's canonical SHA-256 fingerprint for rubric content."""
    return canonical_content_hash(canonical_rubric_content(content))


async def get_or_create_seed_version(
    db: AsyncSession,
    standard: NegotiationStandard,
    content: NegotiationStandardContent | dict[str, Any],
    *,
    admin_id: UUID,
    publication_note: str | None = None,
) -> tuple[NegotiationStandardVersion, bool]:
    """Reuse or create an immutable version under the caller's standard lock.

    The caller must lock ``standard`` before invoking this helper. Content is validated,
    copied into a canonical JSON snapshot, and matched by its SHA-256 fingerprint. The
    savepoint makes a uniqueness race recoverable without poisoning the outer definition
    transaction; a concurrent winner is returned as a reused version.
    """
    normalized = normalize_rubric_content(content)
    snapshot = canonical_rubric_content(normalized)
    content_hash = canonical_content_hash(snapshot)
    existing = (
        await db.execute(
            select(NegotiationStandardVersion).where(
                NegotiationStandardVersion.standard_id == standard.id,
                NegotiationStandardVersion.content_hash == content_hash,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    latest = (
        await db.execute(
            select(NegotiationStandardVersion)
            .where(NegotiationStandardVersion.standard_id == standard.id)
            .order_by(NegotiationStandardVersion.version_number.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    candidate = NegotiationStandardVersion(
        standard_id=standard.id,
        version_number=(latest.version_number + 1) if latest else 1,
        schema_version=normalized.schema_version,
        snapshot=snapshot,
        content_hash=content_hash,
        created_by=admin_id,
        published_by=admin_id,
        publication_note=publication_note,
    )
    try:
        async with db.begin_nested():
            db.add(candidate)
            await db.flush()
    except IntegrityError:
        winner = (
            await db.execute(
                select(NegotiationStandardVersion).where(
                    NegotiationStandardVersion.standard_id == standard.id,
                    NegotiationStandardVersion.content_hash == content_hash,
                )
            )
        ).scalar_one_or_none()
        if winner is not None:
            return winner, False
        raise
    return candidate, True


async def publish_seed_standard(
    db: AsyncSession,
    standard_id: UUID,
    content: NegotiationStandardContent | dict[str, Any],
    *,
    admin_id: UUID,
    publication_note: str | None = None,
) -> tuple[NegotiationStandardVersion, bool]:
    """Validate and publish seed content without committing the caller's transaction.

    The standard row is locked for the complete publication transition. Existing content
    hashes are reused, including the current version, so repeated seed publication is
    idempotent. The caller owns commit and rollback boundaries.
    """
    standard = (
        await db.execute(
            select(NegotiationStandard)
            .where(NegotiationStandard.id == standard_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if standard is None:
        raise SeedStandardConflictError("Rubric standard was not found.")
    if standard.status == "archived":
        raise StandardConflictError("Archived negotiation standards cannot be published")

    normalized = normalize_rubric_content(content)
    validation = validate_standard(normalized, for_publish=True)
    if not validation.valid:
        raise StandardValidationError(validation)

    version, created = await get_or_create_seed_version(
        db,
        standard,
        normalized,
        admin_id=admin_id,
        publication_note=publication_note,
    )
    if not created and standard.current_version_id == version.id:
        return version, False
    standard.current_version_id = version.id
    standard.status = "published"
    standard.updated_by = admin_id
    standard.revision += 1
    return version, True


async def get_current_seed_version(
    db: AsyncSession,
    standard_id: UUID,
) -> NegotiationStandardVersion | None:
    """Return the single currently published version for a seeded standard."""
    standard = (
        await db.execute(select(NegotiationStandard).where(NegotiationStandard.id == standard_id))
    ).scalar_one_or_none()
    if standard is None or standard.status != "published" or standard.current_version_id is None:
        return None
    return (
        await db.execute(
            select(NegotiationStandardVersion).where(
                NegotiationStandardVersion.id == standard.current_version_id,
                NegotiationStandardVersion.standard_id == standard.id,
            )
        )
    ).scalar_one_or_none()


async def seed_rubrics(
    db: AsyncSession,
    *,
    source: str | None = None,
    initiated_by: UUID | None = None,
) -> SeedRunResult:
    """Synchronize configured rubric definitions with independent transactions.

    Source and document failures occur before any database mutation. Each valid
    definition is committed independently so a validation or persistence failure
    cannot contaminate other definitions.
    """
    if not settings.rubric_seed_enabled:
        result = SeedRunResult.new("disabled", disabled=True)
        _audit_seed_result(result, initiated_by)
        return result

    try:
        document = load_rubric_source(source)
    except RubricSourceError as exc:
        result = SeedRunResult.new("unknown")
        result.status = "failed"
        result.warnings.append(exc.message)
        _audit_seed_result(result, initiated_by)
        return result

    result = SeedRunResult.new(document.source_id)
    admin_id = initiated_by or UUID(int=0)
    for definition in document.definitions:
        identity = f"{document.source_id}:{normalize_rubric_identity(definition.name)}"
        try:
            async with db.begin():
                standard, standard_created = await get_or_create_seed_standard(
                    db, definition, source_id=document.source_id, admin_id=admin_id
                )
                content = normalize_rubric_content(definition.draft_content)
                if definition.publish:
                    version, published = await publish_seed_standard(
                        db,
                        standard.id,
                        content,
                        admin_id=admin_id,
                        publication_note=definition.publication_note,
                    )
                else:
                    version, _ = await get_or_create_seed_version(
                        db,
                        standard,
                        content,
                        admin_id=admin_id,
                        publication_note=definition.publication_note,
                    )
                    published = False
                result.add_outcome(
                    DefinitionOutcome(
                        identity=identity,
                        status="created" if standard_created else "reused",
                        version_id=version.id,
                        published=published,
                    ),
                    rubric_created=standard_created,
                    version_created=version.id
                    not in {item.version_id for item in result.definition_outcomes},
                )
        except (ValidationError, StandardValidationError):
            result.add_outcome(
                DefinitionOutcome(
                    identity=identity,
                    status="rejected",
                    error=ClassifiedSeedError(
                        code="validation", message="Rubric definition failed validation."
                    ),
                )
            )
            result.warnings.append("A rubric definition was rejected during validation.")
        except (SeedStandardConflictError, StandardConflictError, IntegrityError):
            result.add_outcome(
                DefinitionOutcome(
                    identity=identity,
                    status="failed",
                    error=ClassifiedSeedError(
                        code="concurrency", message="Rubric synchronization conflict."
                    ),
                )
            )
            result.warnings.append("A rubric definition encountered a synchronization conflict.")
        except (OSError, RuntimeError):
            result.add_outcome(
                DefinitionOutcome(
                    identity=identity,
                    status="failed",
                    error=ClassifiedSeedError(
                        code="persistence", message="Rubric data could not be persisted."
                    ),
                )
            )
            result.warnings.append("A rubric definition could not be persisted.")

    if not result.definition_outcomes:
        result.status = "success"
    elif any(item.status == "failed" for item in result.definition_outcomes):
        result.status = "failed"
    elif result.rejected_definitions:
        result.status = "partial_success"
    else:
        result.status = "success"
    _audit_seed_result(result, initiated_by)
    return result


def _audit_seed_result(result: SeedRunResult, initiated_by: UUID | None) -> None:
    """Emit only safe run metadata and aggregate counters to the audit logger."""
    try:
        log_rubric_seed_completed(
            result.run_id,
            result.source_id,
            initiated_by,
            result.status,
            result.reused_rubrics,
            result.created_rubrics,
            result.reused_versions,
            result.created_versions,
            result.published_versions,
            result.rejected_definitions,
        )
    except Exception:
        # Audit delivery must not hide the safe operational result from the CLI.
        return


DYNAMIC_CAMPAIGN_NAME = "Dynamic Rubric Test Campaign"
DYNAMIC_ADMIN_EMAIL = "admin@cat.ph"
DYNAMIC_AGENT_EMAIL = "agent@cat.ph"
DYNAMIC_SCENARIO_NAME = "Temporary Financial Hardship Payment Arrangement"


def _dynamic_content() -> NegotiationStandardContent:
    """Build the concrete three-block rubric used by the standalone demo seed."""
    blocks = []
    for block_id, category, weight, order in (
        ("call-opening", "Call Opening", 25, 1),
        ("empathy-communication", "Empathy and Communication", 35, 2),
        ("negotiation-resolution", "Negotiation and Resolution", 40, 3),
    ):
        blocks.append(
            {
                "id": block_id,
                "category": category,
                "weight": weight,
                "passing_score": 70,
                "scoring_instructions": (
                    "Score observable performance using the evidence in the session."
                ),
                "positive_behaviors": [
                    {
                        "id": f"{block_id}-positive",
                        "name": "Demonstrates the skill",
                        "description": "Uses clear, respectful, and effective collection practice.",
                        "evidence_instructions": "Use only observable conversation evidence.",
                    }
                ],
                "violations": [],
                "penalties": [],
                "recommendation_guidance": "Reinforce the behavior with targeted coaching.",
                "display_order": order,
            }
        )
    return NegotiationStandardContent(
        schema_version=1,
        overall_passing_score=70,
        blocks=blocks,
    )


async def seed_dynamic_rubric_test(db: AsyncSession) -> SeedRunResult:
    """Create or reuse the concrete campaign, rubric, users, and pending session."""
    result = SeedRunResult.new("bundled-dynamic-rubric")
    async with db.begin():
        admin = (
            await db.execute(
                select(User).where(User.email == DYNAMIC_ADMIN_EMAIL, User.is_active.is_(True))
            )
        ).scalar_one_or_none()
        agent = (
            await db.execute(
                select(User).where(
                    User.email == DYNAMIC_AGENT_EMAIL,
                    User.is_active.is_(True),
                    User.user_type == UserType.AGENT.value,
                )
            )
        ).scalar_one_or_none()
        if admin is None or admin.role != UserRole.ADMIN.value:
            raise RubricSourceError("configuration", "An active administrator account is required.")
        if agent is None:
            raise RubricSourceError(
                "configuration", "An active collection agent account is required."
            )

        campaign = (
            await db.execute(select(Campaign).where(Campaign.name == DYNAMIC_CAMPAIGN_NAME))
        ).scalar_one_or_none()
        if campaign is None:
            campaign = Campaign(
                name=DYNAMIC_CAMPAIGN_NAME,
                description="Dedicated dynamic rubric seed campaign.",
                status=CampaignStatus.ACTIVE.value,
            )
            db.add(campaign)
            await db.flush()
        scenario = (
            await db.execute(select(Scenario).where(Scenario.name == DYNAMIC_SCENARIO_NAME))
        ).scalar_one_or_none()
        if scenario is None:
            scenario = Scenario(
                name=DYNAMIC_SCENARIO_NAME,
                scenario_type="collections",
                description="Temporary financial hardship and payment arrangement.",
                debtor_profile={"hardship": "temporary", "goal": "payment arrangement"},
                is_active=True,
            )
            db.add(scenario)
            await db.flush()
        # Avoid lazy-loading the relationship here: async SQLAlchemy cannot
        # perform implicit IO while evaluating ``campaign.scenarios``.
        scenario_link = await db.execute(
            select(campaign_scenarios.c.scenario_id).where(
                campaign_scenarios.c.campaign_id == campaign.id,
                campaign_scenarios.c.scenario_id == scenario.id,
            )
        )
        if scenario_link.scalar_one_or_none() is None:
            await db.execute(
                campaign_scenarios.insert().values(
                    campaign_id=campaign.id,
                    scenario_id=scenario.id,
                )
            )
        assignment = (
            await db.execute(
                select(CampaignAgent).where(
                    CampaignAgent.campaign_id == campaign.id,
                    CampaignAgent.agent_id == agent.id,
                )
            )
        ).scalar_one_or_none()
        if assignment is None:
            db.add(
                CampaignAgent(
                    campaign_id=campaign.id, agent_id=agent.id, role=CampaignRole.PARTICIPANT.value
                )
            )

        content = _dynamic_content()
        standard = (
            await db.execute(
                select(NegotiationStandard).where(NegotiationStandard.campaign_id == campaign.id)
            )
        ).scalar_one_or_none()
        if standard is None:
            standard = NegotiationStandard(
                campaign_id=campaign.id,
                name="Dynamic Collections Rubric",
                source_id="bundled-dynamic-rubric",
                source_rubric_key="dynamic-collections-rubric",
                description="Dynamic three-block collections evaluation rubric.",
                overall_passing_score=content.overall_passing_score,
                draft_content=canonical_rubric_content(content),
                created_by=admin.id,
                updated_by=admin.id,
            )
            db.add(standard)
            await db.flush()
        version, _ = await publish_seed_standard(
            db,
            standard.id,
            content,
            admin_id=admin.id,
            publication_note="Bundled dynamic rubric seed",
        )
        session = (
            await db.execute(
                select(Session).where(
                    Session.campaign_id == campaign.id,
                    Session.scenario_id == scenario.id,
                    Session.agent_id == agent.id,
                    Session.status == "pending",
                )
            )
        ).scalar_one_or_none()
        if session is None:
            session = Session(
                scenario_id=scenario.id,
                campaign_id=campaign.id,
                agent_id=agent.id,
                status="pending",
                negotiation_standard_version_id=version.id,
            )
            db.add(session)
        elif session.negotiation_standard_version_id != version.id:
            session.negotiation_standard_version_id = version.id
        await db.flush()
        result.campaign_id = campaign.id
        result.scenario_id = scenario.id
        result.agent_id = agent.id
        result.rubric_version_id = version.id
        result.session_id = session.id
        result.status = "success"
    return result


async def run_seed() -> SeedRunResult:
    """Open a database session and execute the configured or bundled seed."""
    async with async_session_factory() as db:
        if settings.rubric_seed_enabled or settings.rubric_source:
            return await seed_rubrics(db)
        return await seed_dynamic_rubric_test(db)


def _failed_run_result(message: str) -> SeedRunResult:
    """Build a safe failed result for an operational database error."""
    result = SeedRunResult.new("unknown")
    result.status = "failed"
    result.warnings.append(message)
    return result


def main() -> int:
    """Run the rubric seeder and emit its safe JSON result to standard output."""
    try:
        result = asyncio.run(run_seed())
    except RubricSourceError as exc:
        result = _failed_run_result(exc.message)
    except SQLAlchemyError:
        result = _failed_run_result("Rubric data could not be persisted.")
    sys.stdout.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
    return int(result.status == "failed")


def _safe_source_identifier(source_id: str) -> str:
    """Return a bounded identifier suitable for results and logs."""
    normalized = " ".join(source_id.split())
    return normalized[:120] if normalized else "unknown"


def _source_error(code: SourceErrorCode, message: str) -> RubricSourceError:
    """Create a source error without including paths, payloads, or credentials."""
    return RubricSourceError(code, message)


def _read_source_bytes(path: Path, max_bytes: int) -> bytes:
    """Read a bounded local JSON file and classify filesystem failures safely."""
    if path.suffix.lower() != ".json":
        raise _source_error("unsupported", "Rubric source must be a local JSON file.")
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise _source_error("unsupported", "Rubric source exceeds the configured size limit.")
        return path.read_bytes()
    except RubricSourceError:
        raise
    except (OSError, ValueError) as exc:
        raise _source_error("unavailable", "Rubric source could not be read.") from exc


def load_rubric_source(
    source: str | None = None,
    *,
    source_id_override: str | None = None,
    max_bytes: int | None = None,
) -> RubricSourceDocument:
    """Load and strictly validate a bounded local JSON rubric source.

    Args:
        source: Local JSON path; defaults to ``CAT_RUBRIC_SOURCE``.
        source_id_override: Optional safe identifier override from configuration.
        max_bytes: Optional byte limit; defaults to ``CAT_RUBRIC_SOURCE_MAX_BYTES``.

    Raises:
        RubricSourceError: If configuration, source format, JSON, or schema is invalid.
    """
    configured_source = source if source is not None else settings.rubric_source
    if not configured_source:
        raise _source_error("configuration", "Rubric source is not configured.")
    if max_bytes is None:
        max_bytes = settings.rubric_source_max_bytes
    if max_bytes < 1:
        raise _source_error("configuration", "Rubric source size limit must be positive.")
    if "://" in configured_source:
        raise _source_error("unsupported", "Only local JSON rubric sources are supported.")

    source_path = Path(configured_source).expanduser()
    raw = _read_source_bytes(source_path, max_bytes)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _source_error("malformed", "Rubric source must contain valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise _source_error("malformed", "Rubric source must be a JSON object.")

    try:
        document = RubricSourceDocument.model_validate(payload)
    except ValidationError as exc:
        raise _source_error(
            "malformed", "Rubric source contains invalid or unsupported definition fields."
        ) from exc
    effective_source_id = source_id_override or settings.rubric_source_id
    if effective_source_id:
        document = document.model_copy(
            update={"source_id": _safe_source_identifier(effective_source_id)}
        )
    return document


__all__ = [
    "ClassifiedSeedError",
    "DefinitionOutcome",
    "RubricDefinition",
    "RubricSourceDocument",
    "RubricSourceError",
    "SeedRunResult",
    "SeedStandardConflictError",
    "canonical_rubric_content",
    "get_current_seed_version",
    "get_or_create_seed_standard",
    "get_or_create_seed_version",
    "load_rubric_source",
    "normalize_rubric_content",
    "normalize_rubric_identity",
    "publish_seed_standard",
    "rubric_content_hash",
    "run_seed",
    "seed_dynamic_rubric_test",
    "seed_rubrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
