"""Unit tests for Pydantic schemas."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.schemas import (
    CoachingReportSchema,
    CompetencyScore,
    DebtorProfileSchema,
    EvaluationCategory,
    EvaluationResult,
    LearningPlanItem,
    LearningPlanSchema,
    MistakeItem,
    PersonaSummary,
    ScenarioListItem,
    ScenarioResponse,
    ScenarioType,
    SessionCreate,
    SessionResponse,
    SessionStatus,
    StrengthItem,
    TranscriptEntry,
    WeaknessItem,
)
from app.schemas.script import (
    ConversationGoal,
    DebtorPersona,
    EmotionalStateRule,
    EscalationConditionEntry,
    ExpectedReplyEntry,
    FinancialSituation,
    PaymentConditionEntry,
    ScriptContract,
    TriggerPhraseEntry,
)


# --- DebtorProfileSchema Tests ---


class TestDebtorProfileSchema:
    """Tests for DebtorProfileSchema validation."""

    def test_valid_profile(self):
        """A complete, valid profile should pass validation."""
        profile = DebtorProfileSchema(
            name="John Doe",
            outstanding_balance=Decimal("5000.00"),
            days_past_due=45,
            personality_profile="anxious",
            conversation_goal="negotiate payment plan",
        )
        assert profile.name == "John Doe"
        assert profile.outstanding_balance == Decimal("5000.00")
        assert profile.days_past_due == 45

    def test_empty_name_rejected(self):
        """An empty name should be rejected."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=30,
                personality_profile="hostile",
                conversation_goal="refuse to pay",
            )

    def test_whitespace_name_rejected(self):
        """A whitespace-only name should be rejected by model validator."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="   ",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=30,
                personality_profile="hostile",
                conversation_goal="refuse to pay",
            )

    def test_zero_balance_rejected(self):
        """A zero balance should be rejected (gt=0)."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="Jane",
                outstanding_balance=Decimal("0"),
                days_past_due=10,
                personality_profile="calm",
                conversation_goal="settle debt",
            )

    def test_negative_balance_rejected(self):
        """A negative balance should be rejected."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="Jane",
                outstanding_balance=Decimal("-500.00"),
                days_past_due=10,
                personality_profile="calm",
                conversation_goal="settle debt",
            )

    def test_zero_days_past_due_rejected(self):
        """Days past due of 0 should be rejected (ge=1)."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="Jane",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=0,
                personality_profile="calm",
                conversation_goal="settle debt",
            )

    def test_whitespace_personality_rejected(self):
        """A whitespace-only personality_profile should be rejected."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="Jane",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=10,
                personality_profile="   ",
                conversation_goal="settle debt",
            )

    def test_whitespace_conversation_goal_rejected(self):
        """A whitespace-only conversation_goal should be rejected."""
        with pytest.raises(Exception):
            DebtorProfileSchema(
                name="Jane",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=10,
                personality_profile="calm",
                conversation_goal="  ",
            )

    def test_minimum_valid_values(self):
        """Minimum boundary values should be accepted."""
        profile = DebtorProfileSchema(
            name="A",
            outstanding_balance=Decimal("0.01"),
            days_past_due=1,
            personality_profile="x",
            conversation_goal="y",
        )
        assert profile.days_past_due == 1
        assert profile.outstanding_balance == Decimal("0.01")


# --- ScenarioType Enum Tests ---


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_all_types_defined(self):
        assert ScenarioType.FINANCIAL_HARDSHIP == "FINANCIAL_HARDSHIP"
        assert ScenarioType.ANGRY_CUSTOMER == "ANGRY_CUSTOMER"
        assert ScenarioType.PAYMENT_EXTENSION == "PAYMENT_EXTENSION"
        assert ScenarioType.BALANCE_DISPUTE == "BALANCE_DISPUTE"

    def test_enum_has_four_members(self):
        assert len(ScenarioType) == 4


# --- ScenarioListItem Tests ---


class TestScenarioListItem:
    """Tests for ScenarioListItem schema."""

    def test_valid_list_item(self):
        item = ScenarioListItem(
            id=uuid.uuid4(),
            name="Financial Hardship",
            scenario_type=ScenarioType.FINANCIAL_HARDSHIP,
        )
        assert item.name == "Financial Hardship"
        assert item.scenario_type == ScenarioType.FINANCIAL_HARDSHIP


# --- ScenarioResponse Tests ---


class TestScenarioResponse:
    """Tests for ScenarioResponse schema."""

    def test_valid_response(self):
        resp = ScenarioResponse(
            id=uuid.uuid4(),
            name="Angry Customer",
            scenario_type=ScenarioType.ANGRY_CUSTOMER,
            description="A frustrated debtor who is angry about the debt.",
            debtor_profile=DebtorProfileSchema(
                name="Bob",
                outstanding_balance=Decimal("2500.00"),
                days_past_due=60,
                personality_profile="hostile",
                conversation_goal="resolve dispute",
            ),
        )
        assert resp.debtor_profile.name == "Bob"


# --- Session Schema Tests ---


class TestSessionSchemas:
    """Tests for session-related schemas."""

    def test_session_create(self):
        sid = uuid.uuid4()
        create = SessionCreate(scenario_id=sid)
        assert create.scenario_id == sid

    def test_session_response(self):
        now = datetime.now(timezone.utc)
        resp = SessionResponse(
            id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            persona=PersonaSummary(
                name="Maria",
                communication_style="evasive",
                emotional_state="defensive",
            ),
            status=SessionStatus.ACTIVE,
            created_at=now,
            ended_at=None,
        )
        assert resp.status == SessionStatus.ACTIVE
        assert resp.persona.name == "Maria"
        assert resp.ended_at is None

    def test_session_response_without_persona(self):
        now = datetime.now(timezone.utc)
        resp = SessionResponse(
            id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            status=SessionStatus.PENDING,
            created_at=now,
        )
        assert resp.persona is None


# --- TranscriptEntry Tests ---


class TestTranscriptEntry:
    """Tests for TranscriptEntry schema."""

    def test_valid_agent_entry(self):
        entry = TranscriptEntry(
            speaker="agent",
            text="Hello, I'm calling about your account.",
            timestamp=datetime.now(timezone.utc),
            sequence_number=0,
        )
        assert entry.speaker == "agent"

    def test_valid_debtor_entry(self):
        entry = TranscriptEntry(
            speaker="debtor",
            text="I know, I'm having trouble paying.",
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
        )
        assert entry.speaker == "debtor"

    def test_invalid_speaker_rejected(self):
        with pytest.raises(Exception):
            TranscriptEntry(
                speaker="system",
                text="Some text",
                timestamp=datetime.now(timezone.utc),
                sequence_number=0,
            )

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            TranscriptEntry(
                speaker="agent",
                text="",
                timestamp=datetime.now(timezone.utc),
                sequence_number=0,
            )


# --- Evaluation Schema Tests ---


class TestEvaluationSchemas:
    """Tests for evaluation-related schemas."""

    def _make_strength(self, category=EvaluationCategory.COMPLIANCE):
        return StrengthItem(
            description="Good compliance check",
            category=category,
            transcript_excerpt="Agent verified identity",
        )

    def _make_weakness(self, category=EvaluationCategory.EMPATHY_COMMUNICATION):
        return WeaknessItem(
            description="Lacked empathy",
            category=category,
            transcript_excerpt="Agent was dismissive",
        )

    def test_competency_score_valid(self):
        score = CompetencyScore(
            category=EvaluationCategory.CALL_OPENING,
            score=85,
            strengths=[self._make_strength(EvaluationCategory.CALL_OPENING)],
            weaknesses=[],
        )
        assert score.score == 85

    def test_competency_score_out_of_range_rejected(self):
        with pytest.raises(Exception):
            CompetencyScore(
                category=EvaluationCategory.COMPLIANCE,
                score=101,
            )

    def test_competency_score_negative_rejected(self):
        with pytest.raises(Exception):
            CompetencyScore(
                category=EvaluationCategory.COMPLIANCE,
                score=-1,
            )

    def test_evaluation_result_valid(self):
        result = EvaluationResult(
            session_id=uuid.uuid4(),
            category_scores=[
                CompetencyScore(category=EvaluationCategory.CALL_OPENING, score=80),
                CompetencyScore(category=EvaluationCategory.COMPLIANCE, score=90),
                CompetencyScore(category=EvaluationCategory.EMPATHY_COMMUNICATION, score=70),
                CompetencyScore(category=EvaluationCategory.NEGOTIATION_RESOLUTION, score=75),
            ],
            overall_score=79.75,
            strengths=[self._make_strength()],
            weaknesses=[self._make_weakness()],
            is_too_short=False,
        )
        assert result.overall_score == 79.75

    def test_evaluation_result_too_many_strengths_rejected(self):
        with pytest.raises(Exception):
            EvaluationResult(
                session_id=uuid.uuid4(),
                category_scores=[],
                overall_score=50.0,
                strengths=[self._make_strength() for _ in range(6)],
                weaknesses=[self._make_weakness()],
            )

    def test_evaluation_result_zero_strengths_rejected(self):
        with pytest.raises(Exception):
            EvaluationResult(
                session_id=uuid.uuid4(),
                category_scores=[],
                overall_score=50.0,
                strengths=[],
                weaknesses=[self._make_weakness()],
            )

    def test_evaluation_result_too_many_weaknesses_rejected(self):
        with pytest.raises(Exception):
            EvaluationResult(
                session_id=uuid.uuid4(),
                category_scores=[],
                overall_score=50.0,
                strengths=[self._make_strength()],
                weaknesses=[self._make_weakness() for _ in range(6)],
            )

    def test_evaluation_result_zero_weaknesses_rejected(self):
        with pytest.raises(Exception):
            EvaluationResult(
                session_id=uuid.uuid4(),
                category_scores=[],
                overall_score=50.0,
                strengths=[self._make_strength()],
                weaknesses=[],
            )


# --- Coaching Schema Tests ---


class TestCoachingSchemas:
    """Tests for coaching-related schemas."""

    def test_mistake_item_valid(self):
        item = MistakeItem(
            transcript_position=3,
            transcript_excerpt="Agent said: pay now or else",
            category=EvaluationCategory.COMPLIANCE,
            explanation="Threatening language violates compliance rules",
            recommended_alternative="I understand this is difficult. Let's discuss options.",
        )
        assert item.transcript_position == 3

    def test_coaching_report_with_mistakes(self):
        mistake = MistakeItem(
            transcript_position=2,
            transcript_excerpt="Agent did not verify identity",
            category=EvaluationCategory.COMPLIANCE,
            explanation="Identity verification is required",
            recommended_alternative="May I verify your name and date of birth?",
        )
        report = CoachingReportSchema(
            session_id=uuid.uuid4(),
            mistakes_by_category={EvaluationCategory.COMPLIANCE: [mistake]},
            total_mistakes=1,
            no_mistakes=False,
        )
        assert report.total_mistakes == 1
        assert not report.no_mistakes

    def test_coaching_report_no_mistakes(self):
        report = CoachingReportSchema(
            session_id=uuid.uuid4(),
            mistakes_by_category={},
            total_mistakes=0,
            no_mistakes=True,
        )
        assert report.no_mistakes
        assert report.total_mistakes == 0


# --- Learning Plan Schema Tests ---


class TestLearningPlanSchemas:
    """Tests for learning plan schemas."""

    def test_learning_plan_with_weaknesses(self):
        plan = LearningPlanSchema(
            session_id=uuid.uuid4(),
            weak_competencies=[
                LearningPlanItem(
                    category=EvaluationCategory.COMPLIANCE,
                    score=55,
                    recommended_scenario="Compliance Fundamentals",
                ),
                LearningPlanItem(
                    category=EvaluationCategory.EMPATHY_COMMUNICATION,
                    score=60,
                    recommended_scenario="Financial Hardship",
                ),
            ],
            all_passing=False,
        )
        assert len(plan.weak_competencies) == 2
        assert not plan.all_passing

    def test_learning_plan_all_passing(self):
        plan = LearningPlanSchema(
            session_id=uuid.uuid4(),
            weak_competencies=[],
            all_passing=True,
        )
        assert plan.all_passing
        assert len(plan.weak_competencies) == 0

    def test_learning_plan_item_score_boundaries(self):
        """Score must be between 0 and 100."""
        with pytest.raises(Exception):
            LearningPlanItem(
                category=EvaluationCategory.CALL_OPENING,
                score=101,
                recommended_scenario="Call Opening Basics",
            )

        with pytest.raises(Exception):
            LearningPlanItem(
                category=EvaluationCategory.CALL_OPENING,
                score=-1,
                recommended_scenario="Call Opening Basics",
            )


# --- SessionStatus Enum Tests ---


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_statuses_defined(self):
        assert SessionStatus.PENDING == "pending"
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.COMPLETED == "completed"
        assert SessionStatus.ERROR == "error"

    def test_enum_has_four_members(self):
        assert len(SessionStatus) == 4


# --- EvaluationCategory Enum Tests ---


class TestEvaluationCategory:
    """Tests for EvaluationCategory enum."""

    def test_all_categories_defined(self):
        assert EvaluationCategory.CALL_OPENING == "call_opening"
        assert EvaluationCategory.COMPLIANCE == "compliance"
        assert EvaluationCategory.EMPATHY_COMMUNICATION == "empathy_communication"
        assert EvaluationCategory.NEGOTIATION_RESOLUTION == "negotiation_resolution"

    def test_enum_has_four_members(self):
        assert len(EvaluationCategory) == 4


# --- ScriptContract Tests ---


def _valid_script_contract_kwargs() -> dict:
    """Build a complete, valid set of ScriptContract constructor kwargs."""
    return dict(
        debtor_persona=DebtorPersona(
            name="Maria Alvarez",
            communication_style="polite but evasive",
            background="Lost her job three months ago and has been juggling bills.",
        ),
        financial_situation=FinancialSituation(
            outstanding_balance=Decimal("3200.50"),
            days_past_due=60,
            reason_for_delinquency="Job loss",
        ),
        opening_response="Hello, who is this calling?",
        expected_replies=[
            ExpectedReplyEntry(
                agent_statement="I'm calling about your overdue account.",
                debtor_reply="I know, I've just been really short on cash lately.",
            ),
        ],
        trigger_phrases=[
            TriggerPhraseEntry(
                phrase="legal action",
                behavior="Debtor becomes anxious and asks for more time.",
            ),
        ],
        emotional_state_rules=[
            EmotionalStateRule(
                trigger="aggressive_tone",
                state_change="increase_defensiveness",
            ),
        ],
        payment_conditions=[
            PaymentConditionEntry(
                condition="offered payment plan under $200/month",
                term="$150/month for 12 months",
                accepted=True,
            ),
        ],
        escalation_conditions=[
            EscalationConditionEntry(
                condition="agent threatens debtor",
                behavior="Debtor hangs up.",
                ends_call=True,
            ),
        ],
        prohibited_responses=["I will never pay this debt."],
        conversation_goal=ConversationGoal(
            target_outcome="Debtor agrees to a payment plan.",
            completion_condition="Debtor verbally commits to a payment amount and date.",
        ),
    )


class TestScriptContract:
    """Unit tests for the ScriptContract schema (Requirements 1.1, 1.9)."""

    def test_fully_valid_contract(self):
        """A contract with all required fields correctly populated should pass validation."""
        contract = ScriptContract(**_valid_script_contract_kwargs())

        assert contract.debtor_persona.name == "Maria Alvarez"
        assert contract.opening_response == "Hello, who is this calling?"
        assert contract.expected_replies[0].debtor_reply.startswith("I know")
        assert contract.conversation_goal.target_outcome == "Debtor agrees to a payment plan."

    def test_missing_required_field_rejected(self):
        """Omitting a required top-level field (conversation_goal) should be rejected."""
        kwargs = _valid_script_contract_kwargs()
        del kwargs["conversation_goal"]

        with pytest.raises(Exception):
            ScriptContract(**kwargs)

    def test_prohibited_expected_conflict_rejected(self):
        """A prohibited_responses entry duplicating an expected_replies debtor_reply is rejected."""
        kwargs = _valid_script_contract_kwargs()
        # Duplicate (trimmed, case-insensitive) the existing expected reply into
        # prohibited_responses to trigger the conflict validator.
        kwargs["prohibited_responses"] = [
            "  I KNOW, I'VE JUST BEEN REALLY SHORT ON CASH LATELY.  "
        ]

        with pytest.raises(Exception):
            ScriptContract(**kwargs)

    def test_unknown_extra_field_rejected(self):
        """An unrecognized extra field at the top level should be rejected (extra='forbid')."""
        kwargs = _valid_script_contract_kwargs()
        kwargs["unexpected_field"] = "should not be allowed"

        with pytest.raises(Exception):
            ScriptContract(**kwargs)
