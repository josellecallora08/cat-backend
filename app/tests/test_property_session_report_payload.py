"""Property-based tests for session report assembly invariants.

Feature: session-report-generation, task 2.3.

Properties:
  1. Determinism: identical stored inputs produce byte-identical payloads/hashes.
  2. No fabricated values: an absent artifact never carries a score or content.
  3. Canonical and legacy evaluation branches are mutually exclusive.
  4. Coaching block identity (rubric_block_id, display_order) is unique.
  5. Transcript sequence numbers are strictly increasing after assembly.
  6. Every recommendation's evidence_sequence_number exists in its block's evidence.
  7. Weighted contributions and scores stay within [0, 100].
  8. Terminal outcomes (too_short, not_applicable) always carry a reason.

Validates: Requirements 1.5, 1.7, 1.10, 1.11, 1.12, 2.5
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.database import Base
from app.models import Evaluation, Scenario, Session, Transcript
from app.schemas import SessionStatus, TranscriptEntry
from app.schemas.rubric_evaluation import (
    CanonicalEvaluationResult,
    RubricCategoryScore,
    RubricEvidence,
)
from app.schemas.session_report import (
    SECTION_REASON_MATRIX,
    CoachingSection,
    EvaluationSection,
    LearningPlanSection,
    LegacyEvaluationResult,
    SessionReportPayload,
    SessionReportSummary,
    TranscriptSection,
)
from app.services.session_report_assembler import (
    assemble_report_payload,
    canonical_serialize,
    compute_content_hash,
)


@pytest.fixture
async def async_db():
    """In-memory SQLite database with foreign keys enabled."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _make_scenario() -> Scenario:
    return Scenario(
        id=uuid.uuid4(),
        name="Test Scenario",
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario",
        debtor_profile={
            "name": "Test Debtor",
            "outstanding_balance": "5000.00",
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "negotiate payment",
        },
        is_active=True,
    )


def _make_session(scenario_id: uuid.UUID) -> Session:
    now = datetime.now(UTC)
    return Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=uuid.uuid4(),
        status="completed",
        persona_context={"name": "Test Persona", "communication_style": "calm", "emotional_state": 3},
        created_at=now - timedelta(minutes=10),
        ended_at=now,
    )


async def _reload_session(async_db: AsyncSession, session_id: uuid.UUID) -> Session:
    stmt = (
        select(Session)
        .options(selectinload(Session.campaign))
        .where(Session.id == session_id)
    )
    result = await async_db.execute(stmt)
    return result.scalar_one()


# --- Strategies ---

utterance_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=80,
).filter(lambda s: s.strip() != "")

speakers = st.sampled_from(["agent", "debtor"])


# --- Property 1: Determinism ---

class TestDeterminism:
    """Property 1: identical stored inputs assemble to byte-identical payloads."""

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_entries=st.integers(min_value=0, max_value=10),
        overall_score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @pytest.mark.asyncio
    async def test_repeated_assembly_yields_identical_hash(
        self, async_db: AsyncSession, num_entries: int, overall_score: float
    ):
        """Assembling the same stored session twice yields the same content hash."""
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()
        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        now = datetime.now(UTC)
        for i in range(num_entries):
            async_db.add(Transcript(
                id=uuid.uuid4(), session_id=session.id,
                speaker="agent" if i % 2 == 0 else "debtor",
                utterance_text=f"utterance {i}", timestamp_ms=now, sequence_number=i,
            ))
        async_db.add(Evaluation(
            id=uuid.uuid4(), session_id=session.id, overall_score=overall_score,
            category_scores=[{"category": "compliance", "score": 80, "strengths": [], "weaknesses": []}],
            strengths=[{"description": "Good opening", "category": "call_opening", "transcript_excerpt": "Hi"}],
            weaknesses=[{"description": "Missed disclosure", "category": "compliance", "transcript_excerpt": "..."}],
            is_too_short=False,
        ))
        await async_db.commit()

        reloaded_1 = await _reload_session(async_db, session.id)
        payload_1 = await assemble_report_payload(async_db, reloaded_1)
        hash_1 = compute_content_hash(payload_1)

        reloaded_2 = await _reload_session(async_db, session.id)
        payload_2 = await assemble_report_payload(async_db, reloaded_2)
        hash_2 = compute_content_hash(payload_2)

        assert hash_1 == hash_2
        await async_db.rollback()


# --- Property 2: No fabricated values ---

class TestNoFabrication:
    """Property 2: absent artifacts never carry a fabricated score or content."""

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(num_transcripts=st.integers(min_value=0, max_value=5))
    @pytest.mark.asyncio
    async def test_absent_artifacts_have_no_content(
        self, async_db: AsyncSession, num_transcripts: int
    ):
        """A session with no evaluation/coaching/learning-plan rows reports absence, not zeros."""
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()
        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        now = datetime.now(UTC)
        for i in range(num_transcripts):
            async_db.add(Transcript(
                id=uuid.uuid4(), session_id=session.id,
                speaker="agent" if i % 2 == 0 else "debtor",
                utterance_text=f"utterance {i}", timestamp_ms=now, sequence_number=i,
            ))
        await async_db.commit()

        reloaded = await _reload_session(async_db, session.id)
        payload = await assemble_report_payload(async_db, reloaded)

        assert payload.evaluation.available is False
        assert payload.evaluation.canonical is None
        assert payload.evaluation.legacy is None
        assert payload.evaluation.mode is None
        assert payload.evaluation.reason

        assert payload.coaching.available is False
        assert payload.coaching.blocks == []
        assert payload.coaching.legacy_mistakes_by_category == {}
        assert payload.coaching.reason

        assert payload.learning_plan.available is False
        assert payload.learning_plan.items == []
        assert payload.learning_plan.reason

        await async_db.rollback()


# --- Property 3: Canonical/legacy mutual exclusivity ---

class TestBranchExclusivity:
    """Property 3: canonical and legacy evaluation data are never both populated."""

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(overall_score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @pytest.mark.asyncio
    async def test_legacy_evaluation_never_has_canonical_data(
        self, async_db: AsyncSession, overall_score: float
    ):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()
        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        async_db.add(Evaluation(
            id=uuid.uuid4(), session_id=session.id, overall_score=overall_score,
            category_scores=[{"category": "compliance", "score": 80, "strengths": [], "weaknesses": []}],
            strengths=[{"description": "Good opening", "category": "call_opening", "transcript_excerpt": "Hi"}],
            weaknesses=[{"description": "Missed disclosure", "category": "compliance", "transcript_excerpt": "..."}],
            is_too_short=False,
        ))
        await async_db.commit()

        reloaded = await _reload_session(async_db, session.id)
        payload = await assemble_report_payload(async_db, reloaded)

        assert payload.evaluation.mode == "legacy"
        assert payload.evaluation.canonical is None
        assert payload.evaluation.legacy is not None
        await async_db.rollback()


# --- Property 5: Transcript ordering ---

class TestTranscriptOrdering:
    """Property 5: assembled transcript entries are strictly increasing by sequence."""

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        sequences=st.lists(
            st.integers(min_value=0, max_value=200), min_size=1, max_size=15, unique=True
        )
    )
    @pytest.mark.asyncio
    async def test_entries_strictly_increasing_regardless_of_insert_order(
        self, async_db: AsyncSession, sequences: list[int]
    ):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()
        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        now = datetime.now(UTC)
        # Insert in reverse to prove assembly sorts, not just relies on insert order.
        for seq in reversed(sequences):
            async_db.add(Transcript(
                id=uuid.uuid4(), session_id=session.id, speaker="agent",
                utterance_text=f"utterance {seq}", timestamp_ms=now, sequence_number=seq,
            ))
        await async_db.commit()

        reloaded = await _reload_session(async_db, session.id)
        payload = await assemble_report_payload(async_db, reloaded)

        result_sequences = [e.sequence_number for e in payload.transcript.entries]
        assert result_sequences == sorted(sequences)
        for a, b in zip(result_sequences, result_sequences[1:]):
            assert a < b
        await async_db.rollback()


# --- Property 8: Terminal outcomes carry a reason ---

class TestTerminalOutcomesHaveReason:
    """Property 8: too_short and not_applicable evaluations always carry a reason."""

    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(is_too_short=st.just(True))
    @pytest.mark.asyncio
    async def test_too_short_has_reason(self, async_db: AsyncSession, is_too_short: bool):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()
        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        async_db.add(Evaluation(
            id=uuid.uuid4(), session_id=session.id, overall_score=0.0,
            category_scores=[], strengths=[], weaknesses=[], is_too_short=is_too_short,
        ))
        await async_db.commit()

        reloaded = await _reload_session(async_db, session.id)
        payload = await assemble_report_payload(async_db, reloaded)

        assert payload.evaluation.mode == "too_short"
        assert payload.evaluation.reason
        assert payload.evaluation.canonical is None
        assert payload.evaluation.legacy is None
        await async_db.rollback()


# --- Deterministic in-memory payload properties for task 2.2 ---


def _memory_summary() -> SessionReportSummary:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return SessionReportSummary(
        session_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        scenario_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        agent_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
        status=SessionStatus.COMPLETED,
        created_at=now,
        ended_at=now,
        duration_seconds=0,
    )


def _memory_transcript(sequences: list[int]) -> TranscriptSection:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    return TranscriptSection(
        available=True,
        entries=[
            TranscriptEntry(
                speaker="agent" if sequence % 2 == 0 else "debtor",
                text=f"utterance {sequence}",
                timestamp=timestamp,
                sequence_number=sequence,
            )
            for sequence in sequences
        ],
    )


def _memory_canonical(
    *,
    sequence: int = 0,
    raw_score: int = 80,
    penalty_total: int = 5,
    weighted_contribution: int = 75,
    evidence: bool = True,
) -> CanonicalEvaluationResult:
    return CanonicalEvaluationResult(
        status="evaluated",
        summary="Stored canonical result",
        categories=[
            RubricCategoryScore(
                rubric_block_id="opening",
                category="Opening",
                raw_score=raw_score,
                penalty_total=penalty_total,
                penalized_score=max(0, raw_score - penalty_total),
                weight=100,
                weighted_contribution=Decimal(weighted_contribution),
                passing_score=70,
                passed=weighted_contribution >= 70,
                evidence=(
                    [
                        RubricEvidence(
                            sequence_number=sequence,
                            speaker="agent" if sequence % 2 == 0 else "debtor",
                            excerpt=f"utterance {sequence}",
                            explanation="Stored evidence",
                        )
                    ]
                    if evidence
                    else []
                ),
                strengths=[],
                violations=[],
                failed_criteria=[],
                recommendation_inputs=[],
            )
        ],
        weighted_total=Decimal(weighted_contribution),
        passing_score=70,
        passed=weighted_contribution >= 70,
        applied_techniques={
            "techniques_used": [],
            "reason_if_empty": "No techniques stored",
        },
        missed_opportunities={
            "missed_techniques": [],
            "reason_if_empty": "No missed opportunities stored",
        },
        recommendations=[],
    )


def _memory_payload(
    *,
    sequences: list[int] | None = None,
    branch: str = "legacy",
    score: int = 80,
    penalty_total: int = 5,
    weighted_contribution: int = 75,
    evaluation: EvaluationSection | None = None,
    coaching: CoachingSection | None = None,
    learning_plan: LearningPlanSection | None = None,
) -> SessionReportPayload:
    sequences = [0] if sequences is None else sequences
    if evaluation is None:
        if branch == "canonical":
            evaluation = EvaluationSection(
                available=True,
                mode="canonical",
                canonical=_memory_canonical(
                    sequence=sequences[0],
                    raw_score=score,
                    penalty_total=penalty_total,
                    weighted_contribution=weighted_contribution,
                ),
                weighted_total=weighted_contribution,
                passing_score=70,
                passed=weighted_contribution >= 70,
            )
        else:
            evaluation = EvaluationSection(
                available=True,
                mode="legacy",
                reason_code="legacy_only",
                legacy=LegacyEvaluationResult(overall_score=float(score)),
            )
    return SessionReportPayload(
        summary=_memory_summary(),
        transcript=_memory_transcript(sequences),
        evaluation=evaluation,
        coaching=coaching or CoachingSection(available=True, mode="legacy"),
        learning_plan=learning_plan or LearningPlanSection(available=True),
    )


valid_sequences = st.lists(
    st.integers(min_value=0, max_value=20), min_size=1, max_size=6, unique=True
).map(sorted)
valid_scores = st.integers(min_value=0, max_value=100)
valid_penalties = st.integers(min_value=0, max_value=100)


class TestDeterministicInMemoryPayloadProperties:
    """Hypothesis properties over already-materialized report contract values."""

    @settings(max_examples=40)
    @given(sequences=valid_sequences, reverse_order=st.booleans(), duplicate=st.booleans())
    def test_transcript_order_and_duplicate_invariants(
        self, sequences: list[int], reverse_order: bool, duplicate: bool
    ):
        """Transcript validation accepts only strictly increasing unique sequences."""
        ordered = list(reversed(sequences)) if reverse_order else list(sequences)
        if duplicate:
            ordered.append(sequences[0])
        if ordered != sorted(ordered) or len(ordered) != len(set(ordered)):
            with pytest.raises(ValidationError):
                _memory_transcript(ordered)
        else:
            section = _memory_transcript(ordered)
            assert [entry.sequence_number for entry in section.entries] == ordered

    @settings(max_examples=35)
    @given(
        score=valid_scores,
        penalty_total=valid_penalties,
        weighted_contribution=valid_scores,
    )
    def test_canonical_score_penalty_and_weight_bounds_are_preserved(
        self, score: int, penalty_total: int, weighted_contribution: int
    ):
        """Generated in-range stored scores remain in-range without recomputation."""
        payload = _memory_payload(
            branch="canonical",
            score=score,
            penalty_total=penalty_total,
            weighted_contribution=weighted_contribution,
        )
        category = payload.evaluation.canonical.categories[0]
        assert category.raw_score == score
        assert category.penalty_total == penalty_total
        assert 0 <= category.weighted_contribution <= 100
        assert payload.evaluation.weighted_total == weighted_contribution

    @settings(max_examples=25)
    @given(score=st.one_of(st.integers(max_value=-1), st.integers(min_value=101)))
    def test_out_of_range_scores_are_rejected(self, score: int):
        """Score bounds reject generated values outside the stored contract."""
        with pytest.raises(ValidationError):
            _memory_payload(branch="canonical", score=score)

    @settings(max_examples=30)
    @given(branch=st.sampled_from(["canonical", "legacy"]))
    def test_evaluation_branch_exclusivity(self, branch: str):
        """Each generated evaluation contains exactly one evaluation branch."""
        evaluation = _memory_payload(branch=branch).evaluation
        if branch == "canonical":
            assert evaluation.mode == "canonical"
            assert evaluation.canonical is not None
            assert evaluation.legacy is None
        else:
            assert evaluation.mode == "legacy"
            assert evaluation.legacy is not None
            assert evaluation.canonical is None

    @settings(max_examples=20)
    @given(include_legacy=st.booleans())
    def test_coaching_branch_exclusivity(self, include_legacy: bool):
        """Canonical coaching cannot be combined with legacy coaching content."""
        canonical_block = {
            "rubric_block_id": "opening",
            "block_name": "Opening",
            "display_order": 0,
            "recommendations": [],
        }
        kwargs = {
            "available": True,
            "mode": "canonical",
            "blocks": [canonical_block],
            "legacy_mistakes_by_category": (
                {"compliance": []} if include_legacy else {}
            ),
        }
        if include_legacy:
            with pytest.raises(ValidationError):
                CoachingSection(**kwargs)
        else:
            section = CoachingSection(**kwargs)
            assert section.mode == "canonical"
            assert section.legacy_mistakes_by_category == {}

    @settings(max_examples=30)
    @given(
        sequences=valid_sequences,
        reference=st.integers(min_value=0, max_value=20),
    )
    def test_evidence_references_resolve_to_transcript_sequences(
        self, sequences: list[int], reference: int
    ):
        """Canonical evidence either resolves or is rejected, never substituted."""
        try:
            payload = _memory_payload(
                branch="canonical",
                sequences=sequences,
                evaluation=EvaluationSection(
                    available=True,
                    mode="canonical",
                    canonical=_memory_canonical(sequence=reference),
                    weighted_total=75,
                    passing_score=70,
                    passed=True,
                ),
            )
        except ValueError:
            payload = None
        if reference in sequences:
            assert payload is not None
            assert (
                payload.evaluation.canonical.categories[0].evidence[0].sequence_number
                == reference
            )
        else:
            assert payload is None

    @settings(max_examples=20)
    @given(missing_section=st.sampled_from(["evaluation", "coaching", "learning_plan"]))
    def test_missing_artifacts_have_typed_reasons_and_empty_content(
        self, missing_section: str
    ):
        """Unavailable sections carry reasons and do not fabricate content."""
        sections = {
            "evaluation": EvaluationSection(
                available=False,
                reason="Not stored",
                reason_code="artifact_missing",
            ),
            "coaching": CoachingSection(
                available=False,
                reason="Not stored",
                reason_code="artifact_missing",
            ),
            "learning_plan": LearningPlanSection(
                available=False,
                reason="Not stored",
                reason_code="artifact_missing",
            ),
        }
        payload = _memory_payload(**{missing_section: sections[missing_section]})
        section = getattr(payload, missing_section)
        assert section.available is False
        assert section.reason_code == "artifact_missing"
        content = getattr(section, "items", None)
        if content is None:
            content = getattr(section, "blocks", None)
        if content is None:
            assert section.canonical is None and section.legacy is None
        else:
            assert content == []

    @settings(max_examples=25)
    @given(mode=st.sampled_from(["not_applicable", "too_short"]))
    def test_terminal_modes_have_reason_and_no_scored_outcome(self, mode: str):
        """Terminal evaluation modes exclude misleading score/pass fields."""
        reason = "not_applicable" if mode == "not_applicable" else "session_too_short"
        evaluation = EvaluationSection(
            available=True,
            mode=mode,
            reason="Terminal stored result",
            reason_code=reason,
        )
        payload = _memory_payload(evaluation=evaluation)
        assert payload.evaluation.reason_code == reason
        assert payload.evaluation.canonical is None
        assert payload.evaluation.legacy is None
        assert payload.evaluation.weighted_total is None
        assert payload.evaluation.passing_score is None
        assert payload.evaluation.passed is None

    @settings(max_examples=20)
    @given(mode=st.sampled_from(["not_applicable", "too_short"]))
    def test_terminal_modes_reject_scored_outcomes(self, mode: str):
        reason = "not_applicable" if mode == "not_applicable" else "session_too_short"
        with pytest.raises(ValidationError):
            EvaluationSection(
                available=True,
                mode=mode,
                reason="Terminal stored result",
                reason_code=reason,
                weighted_total=0,
            )

    @settings(max_examples=30)
    @given(
        section_name=st.sampled_from(tuple(SECTION_REASON_MATRIX)),
        reason_code=st.sampled_from(
            ["artifact_missing", "empty_transcript", "not_applicable", "session_too_short", "legacy_only", "no_evidence", "no_coaching", "no_learning_plan"]
        ),
    )
    def test_reason_matrix_rejects_cross_section_reasons(
        self, section_name: str, reason_code: str
    ):
        """A section cannot silently accept a reason owned by another section."""
        allowed = SECTION_REASON_MATRIX[section_name]
        if reason_code in allowed:
            return
        section_types = {
            "transcript": TranscriptSection,
            "evaluation": EvaluationSection,
            "coaching": CoachingSection,
            "learning_plan": LearningPlanSection,
        }
        with pytest.raises(ValidationError):
            section_types[section_name](
                available=False,
                reason="Invalid cross-section reason",
                reason_code=reason_code,
            )

    @settings(max_examples=25)
    @given(extra_sequence=st.integers(min_value=1, max_value=20))
    def test_adding_transcript_does_not_recompute_evaluation(self, extra_sequence: int):
        """Adding an unreferenced transcript entry changes only transcript content."""
        if extra_sequence == 0:
            return
        base = _memory_payload(branch="canonical", sequences=[0])
        expanded = _memory_payload(branch="canonical", sequences=[0, extra_sequence])
        base_data = base.model_dump(mode="json")
        expanded_data = expanded.model_dump(mode="json")
        base_transcript = base_data.pop("transcript")
        expanded_transcript = expanded_data.pop("transcript")
        assert expanded_data == base_data
        assert expanded_transcript["entries"][:-1] == base_transcript["entries"]
        assert expanded_transcript["entries"][-1]["sequence_number"] == extra_sequence
        assert canonical_serialize(base) != canonical_serialize(expanded)

    @settings(max_examples=25)
    @given(key_order=st.permutations(("summary", "transcript", "evaluation", "coaching", "learning_plan")))
    def test_stable_serialization_ignores_mapping_key_order(self, key_order: tuple[str, ...]):
        """Equivalent stable inputs serialize to identical bytes and hashes."""
        payload = _memory_payload(branch="canonical", sequences=[0])
        data = payload.model_dump(mode="json")
        reordered = {key: data[key] for key in key_order}
        assert canonical_serialize(data) == canonical_serialize(reordered)
        assert compute_content_hash(data) == compute_content_hash(reordered)
