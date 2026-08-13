"""Deterministic assembly of a SessionReportPayload from stored artifacts.

The assembler is deliberately a projection boundary.  It validates the
identity and references of persisted artifacts, but never invokes any
business-result generator (scoring, penalties, evidence selection,
recommendations, coaching grouping, or learning-plan ranking).
"""

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models import CoachingReport, Evaluation, LearningPlan, Session, Transcript
from app.schemas import (
    EvaluationCategory,
    LearningPlanItem,
    MistakeItem,
    PersonaSummary,
    SessionStatus,
    StrengthItem,
    TranscriptEntry,
    WeaknessItem,
)
from app.schemas.rubric_evaluation import CanonicalEvaluationResult, RubricCoaching
from app.schemas.session_report import (
    CoachingSection,
    EvaluationSection,
    LearningPlanSection,
    LegacyEvaluationResult,
    ReportReasonCode,
    SessionReportPayload,
    SessionReportSummary,
    TranscriptSection,
)


def _require_uuid(value: object, name: str) -> UUID:
    if not isinstance(value, UUID):
        raise ValueError(f"{name} is missing or invalid")
    return value


def _require_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} is missing or invalid")
    # SQLite returns timezone-aware ORM values as naive datetimes.  Normalize
    # that dialect representation without accepting missing/non-datetime data.
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value


def _build_persona_summary(persona_context: dict | None) -> PersonaSummary | None:
    """Mirror app.api.sessions._build_persona_summary exactly."""
    if not persona_context:
        return None
    if not isinstance(persona_context, dict):
        raise ValueError("persona_context is invalid")
    return PersonaSummary(
        name=persona_context.get("name", ""),
        communication_style=persona_context.get("communication_style", ""),
        emotional_state=str(persona_context.get("emotional_state", "")),
    )


def _build_summary(session: Session) -> SessionReportSummary:
    """Build summary identity without fabricating timestamps or duration."""
    session_id = _require_uuid(getattr(session, "id", None), "session_id")
    scenario_id = _require_uuid(getattr(session, "scenario_id", None), "scenario_id")
    agent_id = _require_uuid(getattr(session, "agent_id", None), "agent_id")
    created_at = _require_timestamp(getattr(session, "created_at", None), "created_at")
    ended_at = getattr(session, "ended_at", None)
    if ended_at is not None:
        ended_at = _require_timestamp(ended_at, "ended_at")
        if ended_at < created_at:
            raise ValueError("ended_at must not precede created_at")

    try:
        status = SessionStatus(getattr(session, "status", None))
    except (TypeError, ValueError) as exc:
        raise ValueError("session status is invalid") from exc
    if status == SessionStatus.COMPLETED and ended_at is None:
        raise ValueError("completed sessions require ended_at")

    version = getattr(session, "negotiation_standard_version", None)
    standard = version.standard if version is not None else None
    campaign = getattr(session, "campaign", None)
    campaign_id = getattr(session, "campaign_id", None)
    if campaign_id is not None:
        _require_uuid(campaign_id, "campaign_id")
    if campaign is not None and getattr(campaign, "id", campaign_id) != campaign_id:
        raise ValueError("campaign identity does not match the session")

    standard_version_id = getattr(session, "negotiation_standard_version_id", None)
    if standard_version_id is not None:
        _require_uuid(standard_version_id, "standard_version_id")
    if version is not None and getattr(version, "id", standard_version_id) != standard_version_id:
        raise ValueError("standard version identity does not match the session")
    standard_id = getattr(standard, "id", None) if standard is not None else None
    if standard_id is not None:
        _require_uuid(standard_id, "standard_id")

    duration_seconds = (
        (ended_at - created_at).total_seconds() if ended_at is not None else None
    )
    if duration_seconds is not None and duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")

    return SessionReportSummary(
        session_id=session_id,
        scenario_id=scenario_id,
        agent_id=agent_id,
        campaign_id=campaign_id,
        campaign_name=getattr(campaign, "name", None),
        persona=_build_persona_summary(session.persona_context),
        status=status,
        created_at=created_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        standard_id=standard_id,
        standard_version_id=standard_version_id,
        standard_version_number=(
            version.version_number if version is not None else None
        ),
        standard_name=standard.name if standard is not None else None,
    )


def _transcript_index(
    transcripts: list[Transcript], expected_session_id: UUID | None = None
) -> dict[int, TranscriptEntry]:
    """Validate persisted transcript identity and return it indexed by sequence."""
    seen_ids: set[UUID] = set()
    seen_identity: set[tuple[object, ...]] = set()
    entries: list[TranscriptEntry] = []
    for transcript in transcripts:
        transcript_id = _require_uuid(getattr(transcript, "id", None), "transcript_id")
        session_id = getattr(transcript, "session_id", None)
        if expected_session_id is not None and session_id != expected_session_id:
            raise ValueError("transcript session identity does not match the report session")
        if transcript_id in seen_ids:
            raise ValueError("duplicate transcript identity")
        seen_ids.add(transcript_id)
        timestamp = _require_timestamp(getattr(transcript, "timestamp_ms", None), "transcript timestamp")
        text = getattr(transcript, "utterance_text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("transcript text is missing or invalid")
        sequence = getattr(transcript, "sequence_number", None)
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError("transcript sequence number is invalid")
        speaker = getattr(transcript, "speaker", None)
        if speaker not in {"agent", "debtor"}:
            raise ValueError("transcript speaker is invalid")
        identity = (speaker, text, timestamp, sequence)
        if identity in seen_identity:
            raise ValueError("duplicate transcript entry identity")
        seen_identity.add(identity)
        entries.append(
            TranscriptEntry(
                speaker=speaker,
                text=text,
                timestamp=timestamp,
                sequence_number=sequence,
            )
        )

    entries.sort(key=lambda entry: entry.sequence_number)
    sequences = [entry.sequence_number for entry in entries]
    if any(left >= right for left, right in zip(sequences, sequences[1:])):
        raise ValueError("transcript sequence numbers must be strictly increasing")
    return {entry.sequence_number: entry for entry in entries}


def _build_transcript_section(
    transcripts: list[Transcript], expected_session_id: UUID | None = None
) -> TranscriptSection:
    """Build a strictly ordered transcript section from stored rows."""
    indexed = _transcript_index(transcripts, expected_session_id)
    entries = [indexed[sequence] for sequence in sorted(indexed)]
    if not entries:
        return TranscriptSection(
            available=True,
            reason_code=ReportReasonCode.EMPTY_TRANSCRIPT,
        )
    return TranscriptSection(available=True, entries=entries)


def _validate_evidence_reference(
    sequence_number: int,
    transcript_by_sequence: dict[int, TranscriptEntry],
    *,
    speaker: str | None = None,
    excerpt: str | None = None,
) -> None:
    entry = transcript_by_sequence.get(sequence_number)
    if entry is None:
        raise ValueError("artifact contains an invalid transcript cross-reference")
    if speaker is not None and speaker != entry.speaker:
        raise ValueError("artifact evidence speaker does not match the transcript")
    if excerpt is not None and excerpt not in entry.text:
        raise ValueError("artifact evidence excerpt does not match the transcript")


def _validate_pinned_version(obj: object, session: Session) -> None:
    """Reject artifact metadata that points at a different pinned standard."""
    expected_id = getattr(session, "negotiation_standard_version_id", None)
    version_id = getattr(obj, "standard_version_id", None)
    if version_id is not None and expected_id is not None and version_id != expected_id:
        raise ValueError("artifact standard version does not match the pinned session standard")
    expected_version = getattr(getattr(session, "negotiation_standard_version", None), "version_number", None)
    version_number = getattr(obj, "standard_version_number", None)
    if version_number is not None and expected_version is not None and version_number != expected_version:
        raise ValueError("artifact standard version number does not match the pinned session standard")


def _validate_canonical_references(
    canonical: CanonicalEvaluationResult,
    transcript_by_sequence: dict[int, TranscriptEntry],
) -> None:
    block_ids: set[str] = set()
    recommendation_inputs: set[tuple[str, int]] = set()
    recommendations: set[tuple[str, str, int]] = set()
    for category in canonical.categories:
        if category.rubric_block_id in block_ids:
            raise ValueError("duplicate canonical rubric block identity")
        block_ids.add(category.rubric_block_id)
        failed_criteria = list(category.failed_criteria)
        if len(failed_criteria) != len(set(failed_criteria)):
            raise ValueError("duplicate canonical criterion identity")
        evidence_ids: set[tuple[int, str, str]] = set()
        for evidence in category.evidence:
            _validate_evidence_reference(
                evidence.sequence_number,
                transcript_by_sequence,
                speaker=evidence.speaker,
                excerpt=evidence.excerpt,
            )
            evidence_id = (evidence.sequence_number, evidence.speaker, evidence.excerpt)
            if evidence_id in evidence_ids:
                raise ValueError("duplicate canonical evidence identity")
            evidence_ids.add(evidence_id)
        for strength in category.strengths:
            for sequence in strength.evidence_sequence_numbers:
                _validate_evidence_reference(sequence, transcript_by_sequence)
        for violation in category.violations:
            for sequence in violation.evidence_sequence_numbers:
                _validate_evidence_reference(sequence, transcript_by_sequence)
        for item in category.recommendation_inputs:
            _validate_evidence_reference(item.transcript_sequence_number, transcript_by_sequence)
            identity = (item.criterion_id, item.transcript_sequence_number)
            if identity in recommendation_inputs:
                raise ValueError("duplicate canonical recommendation-input identity")
            recommendation_inputs.add(identity)
    for recommendation in canonical.recommendations:
        if recommendation.rubric_block_id not in block_ids:
            raise ValueError("recommendation references an unknown rubric block")
        _validate_evidence_reference(
            recommendation.evidence_sequence_number,
            transcript_by_sequence,
            speaker=recommendation.source_speaker,
            excerpt=recommendation.source_excerpt,
        )
        identity = (
            recommendation.rubric_block_id,
            recommendation.criterion_id,
            recommendation.evidence_sequence_number,
        )
        if identity in recommendations:
            raise ValueError("duplicate canonical recommendation identity")
        recommendations.add(identity)
    for technique in canonical.applied_techniques.techniques_used:
        for sequence in technique.evidence_sequence_numbers:
            _validate_evidence_reference(sequence, transcript_by_sequence)


def _build_evaluation_section(
    evaluation: Evaluation | None,
    session: Session | None = None,
    transcripts: list[Transcript] | None = None,
) -> EvaluationSection:
    """Build canonical, legacy, or terminal evaluation without recomputation."""
    if evaluation is None:
        return EvaluationSection(
            available=False,
            reason="No evaluation recorded for this session",
            reason_code=ReportReasonCode.ARTIFACT_MISSING,
        )
    if session is not None and getattr(evaluation, "session_id", session.id) != session.id:
        raise ValueError("evaluation session identity does not match the report session")
    if session is not None:
        expected_version_id = getattr(session, "negotiation_standard_version_id", None)
        evaluation_version_id = getattr(evaluation, "negotiation_standard_version_id", None)
        if evaluation_version_id is not None and expected_version_id is not None and evaluation_version_id != expected_version_id:
            raise ValueError("evaluation standard version does not match the pinned session standard")

    rubric_result = evaluation.rubric_result
    if rubric_result is not None and not isinstance(rubric_result, dict):
        raise ValueError("rubric result is invalid")
    categories = rubric_result.get("categories") if rubric_result is not None else None
    if rubric_result is not None and not isinstance(categories, list):
        raise ValueError("rubric result categories are invalid")
    has_canonical = bool(categories)
    transcript_by_sequence = _transcript_index(transcripts or []) if transcripts is not None else {}

    version = getattr(evaluation, "negotiation_standard_version", None)
    standard_version_number = getattr(version, "version_number", None)
    if session is not None and version is not None:
        _validate_pinned_version(
            type("VersionReference", (), {
                "standard_version_id": getattr(version, "id", None),
                "standard_version_number": standard_version_number,
            })(),
            session,
        )

    if evaluation.is_too_short:
        if has_canonical or evaluation.category_scores or any(
            value is not None for value in (evaluation.weighted_total, evaluation.passing_score, evaluation.passed)
        ):
            raise ValueError("too-short evaluation contains scored data")
        return EvaluationSection(
            available=True,
            mode="too_short",
            reason="The session transcript was too short to evaluate",
            reason_code=ReportReasonCode.SESSION_TOO_SHORT,
        )

    if has_canonical:
        canonical = CanonicalEvaluationResult.model_validate(rubric_result)
        if canonical.status == "evaluated" and any(category.raw_score is None for category in canonical.categories):
            raise ValueError("evaluated canonical results require scores")
        if canonical.status == "not_applicable" and any(category.raw_score is not None for category in canonical.categories):
            raise ValueError("not-applicable canonical results must not contain scores")
        if transcripts is not None:
            _validate_canonical_references(canonical, transcript_by_sequence)
        if canonical.status == "not_applicable":
            if any(value is not None for value in (evaluation.weighted_total, evaluation.passing_score, evaluation.passed)):
                raise ValueError("not-applicable evaluation contains scored outcome fields")
            return EvaluationSection(
                available=True,
                mode="not_applicable",
                reason=canonical.summary,
                reason_code=ReportReasonCode.NOT_APPLICABLE,
                canonical=canonical,
                standard_version_number=standard_version_number,
            )
        return EvaluationSection(
            available=True,
            mode="canonical",
            canonical=canonical,
            reason_code=(
                ReportReasonCode.NO_EVIDENCE
                if not any(category.evidence for category in canonical.categories)
                else None
            ),
            weighted_total=evaluation.weighted_total,
            passing_score=evaluation.passing_score,
            passed=evaluation.passed,
            standard_version_number=standard_version_number,
        )

    category_scores = evaluation.category_scores or []
    category_names = [item.get("category") for item in category_scores if isinstance(item, dict)]
    if len(category_names) != len(set(category_names)):
        raise ValueError("duplicate legacy evaluation category identity")
    legacy = LegacyEvaluationResult(
        category_scores=[
            {**cs, "category": EvaluationCategory(cs["category"])}
            for cs in category_scores
        ],
        overall_score=evaluation.overall_score,
        strengths=[
            StrengthItem(**{**s, "category": EvaluationCategory(s["category"])})
            for s in (evaluation.strengths or [])
        ],
        weaknesses=[
            WeaknessItem(**{**w, "category": EvaluationCategory(w["category"])})
            for w in (evaluation.weaknesses or [])
        ],
    )
    return EvaluationSection(
        available=True,
        mode="legacy",
        reason_code=ReportReasonCode.LEGACY_ONLY,
        legacy=legacy,
        weighted_total=evaluation.weighted_total,
        passing_score=evaluation.passing_score,
        passed=evaluation.passed,
        standard_version_number=standard_version_number,
    )


def _build_coaching_section(
    report: CoachingReport | None,
    session: Session | None = None,
    transcripts: list[Transcript] | None = None,
) -> CoachingSection:
    """Build canonical coaching once, suppressing legacy duplicates."""
    if report is None:
        return CoachingSection(
            available=False,
            reason="No coaching report recorded for this session",
            reason_code=ReportReasonCode.ARTIFACT_MISSING,
        )
    if session is not None and getattr(report, "session_id", session.id) != session.id:
        raise ValueError("coaching session identity does not match the report session")
    raw_mistakes = report.mistakes_by_category or {}
    if not isinstance(raw_mistakes, dict):
        raise ValueError("coaching artifact is invalid")
    transcript_by_sequence = _transcript_index(transcripts or []) if transcripts is not None else {}
    rubric_coaching = None
    if raw_mistakes.get("_rubric_coaching") is not None:
        rubric_coaching = RubricCoaching.model_validate(raw_mistakes["_rubric_coaching"])
        if session is not None:
            _validate_pinned_version(rubric_coaching, session)

    has_canonical_coaching = bool(
        rubric_coaching is not None
        or raw_mistakes.get("_rubric_recommendations")
        or raw_mistakes.get("_rubric_recommendations_by_block")
    )
    if has_canonical_coaching:
        blocks = rubric_coaching.blocks if rubric_coaching is not None else []
        block_ids: set[str] = set()
        criteria: set[tuple[str, str]] = set()
        for block in blocks:
            if block.rubric_block_id in block_ids:
                raise ValueError("duplicate coaching rubric block identity")
            block_ids.add(block.rubric_block_id)
            for recommendation in block.recommendations:
                if recommendation.rubric_block_id != block.rubric_block_id:
                    raise ValueError("coaching recommendation references the wrong rubric block")
                if session is not None:
                    _validate_pinned_version(recommendation, session)
                identity = (block.rubric_block_id, recommendation.criterion_id)
                if identity in criteria:
                    raise ValueError("duplicate coaching criterion identity")
                criteria.add(identity)
                if transcripts is not None:
                    _validate_evidence_reference(
                        recommendation.evidence_sequence_number,
                        transcript_by_sequence,
                        speaker=recommendation.source_speaker,
                        excerpt=recommendation.source_excerpt,
                    )
        return CoachingSection(
            available=True,
            mode="canonical",
            blocks=sorted(blocks, key=lambda block: (block.display_order, block.rubric_block_id)),
            total_mistakes=report.total_mistakes,
            no_mistakes=report.no_mistakes,
        )

    legacy_mistakes: dict[str, list[MistakeItem]] = {}
    legacy_identities: set[tuple[str, int, str]] = set()
    for category_key, mistakes in raw_mistakes.items():
        if category_key.startswith("_"):
            continue
        try:
            EvaluationCategory(category_key)
        except ValueError:
            continue
        if not isinstance(mistakes, list):
            raise ValueError("legacy coaching mistakes are invalid")
        parsed_mistakes: list[MistakeItem] = []
        for mistake in mistakes:
            parsed = MistakeItem(**mistake)
            if transcripts is not None:
                _validate_evidence_reference(parsed.transcript_position, transcript_by_sequence)
            identity = (category_key, parsed.transcript_position, parsed.transcript_excerpt)
            if identity in legacy_identities:
                raise ValueError("duplicate legacy coaching evidence identity")
            legacy_identities.add(identity)
            parsed_mistakes.append(parsed)
        legacy_mistakes[category_key] = parsed_mistakes

    return CoachingSection(
        available=True,
        mode="legacy",
        legacy_mistakes_by_category=legacy_mistakes,
        total_mistakes=report.total_mistakes,
        no_mistakes=report.no_mistakes,
    )


def _build_learning_plan_section(
    plan: LearningPlan | None,
    session: Session | None = None,
    evaluation: Evaluation | None = None,
) -> LearningPlanSection:
    """Project stored learning-plan items and gate scenario references."""
    if plan is None:
        return LearningPlanSection(
            available=False,
            reason="No learning plan recorded for this session",
            reason_code=ReportReasonCode.ARTIFACT_MISSING,
        )
    if session is not None:
        if getattr(plan, "session_id", session.id) != session.id:
            raise ValueError("learning-plan session identity does not match the report session")
        if getattr(plan, "agent_id", session.agent_id) != session.agent_id:
            raise ValueError("learning-plan agent identity does not match the report session")

    items: list[LearningPlanItem] = []
    identities: set[tuple[str, str]] = set()
    canonical_blocks: dict[str, set[str]] = {}
    if evaluation is not None and evaluation.rubric_result and evaluation.rubric_result.get("categories"):
        for category in evaluation.rubric_result["categories"]:
            block_id = category.get("rubric_block_id")
            canonical_blocks[block_id] = set(category.get("failed_criteria") or [])

    for raw_item in plan.weak_competencies or []:
        item = LearningPlanItem(**raw_item)
        if item.scenario_id is not None:
            if session is None or item.scenario_id != session.scenario_id:
                raise ValueError("learning-plan scenario reference is unauthorized or mismatched")
        if (item.rubric_block_id is None) != (item.criterion_id is None):
            raise ValueError("learning-plan rubric block and criterion must be paired")
        if item.rubric_block_id is not None:
            if not item.rubric_block_id.strip() or not item.criterion_id or not item.criterion_id.strip():
                raise ValueError("learning-plan rubric identity is invalid")
            identity = (item.rubric_block_id, item.criterion_id)
            if identity in identities:
                raise ValueError("duplicate learning-plan criterion identity")
            identities.add(identity)
            if canonical_blocks and (
                item.rubric_block_id not in canonical_blocks
                or item.criterion_id not in canonical_blocks[item.rubric_block_id]
            ):
                raise ValueError("learning-plan item references an unknown canonical criterion")
        items.append(item)

    return LearningPlanSection(available=True, items=items, all_passing=plan.all_passing)


async def _load_artifacts(
    db: AsyncSession, session_id: UUID
) -> tuple[list[Transcript], Evaluation | None, CoachingReport | None, LearningPlan | None]:
    """Load report artifacts with exactly four bounded artifact queries."""
    transcript_stmt = (
        select(Transcript)
        .where(Transcript.session_id == session_id)
        .order_by(Transcript.sequence_number.asc())
    )
    evaluation_stmt = (
        select(Evaluation)
        .options(joinedload(Evaluation.negotiation_standard_version))
        .where(Evaluation.session_id == session_id)
    )
    coaching_stmt = select(CoachingReport).where(CoachingReport.session_id == session_id)
    learning_plan_stmt = select(LearningPlan).where(LearningPlan.session_id == session_id)

    transcripts = (await db.execute(transcript_stmt)).scalars().all()
    evaluation = (await db.execute(evaluation_stmt)).scalar_one_or_none()
    coaching = (await db.execute(coaching_stmt)).scalar_one_or_none()
    learning_plan = (await db.execute(learning_plan_stmt)).scalar_one_or_none()
    return list(transcripts), evaluation, coaching, learning_plan


async def assemble_report_payload(db: AsyncSession, session: Session) -> SessionReportPayload:
    """Assemble and structurally validate a report from stored artifacts only."""
    summary = _build_summary(session)
    transcripts, evaluation, coaching, learning_plan = await _load_artifacts(db, summary.session_id)
    transcript = _build_transcript_section(transcripts, summary.session_id)
    evaluation_section = _build_evaluation_section(evaluation, session, transcripts)
    coaching_section = _build_coaching_section(coaching, session, transcripts)
    learning_plan_section = _build_learning_plan_section(learning_plan, session, evaluation)
    return SessionReportPayload(
        summary=summary,
        transcript=transcript,
        evaluation=evaluation_section,
        coaching=coaching_section,
        learning_plan=learning_plan_section,
    )


def canonical_serialize(payload: SessionReportPayload) -> str:
    """Serialize a validated payload with stable keys and separators."""
    if not isinstance(payload, SessionReportPayload):
        payload = SessionReportPayload.model_validate(payload)
    data = payload.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_content_hash(payload: SessionReportPayload) -> str:
    """Compute the SHA-256 digest of canonical report serialization."""
    return hashlib.sha256(canonical_serialize(payload).encode("utf-8")).hexdigest()
