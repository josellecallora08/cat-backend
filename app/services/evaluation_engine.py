"""Evaluation Engine service for scoring agent performance.

Analyzes completed transcripts and produces weighted competency scores
across four categories: Call Opening, Compliance, Empathy & Communication,
and Negotiation & Resolution.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

import json
import logging
from uuid import UUID

from app.schemas import (
    CompetencyScore,
    EvaluationCategory,
    EvaluationResult,
    StrengthItem,
    WeaknessItem,
)
from app.services.db_retry import retry_db_operation
from app.services.llm_service import LLMMessage, LLMServiceProtocol

logger = logging.getLogger(__name__)


# Category weights as defined in the design document
CATEGORY_WEIGHTS: dict[EvaluationCategory, float] = {
    EvaluationCategory.CALL_OPENING: 0.20,
    EvaluationCategory.COMPLIANCE: 0.30,
    EvaluationCategory.EMPATHY_COMMUNICATION: 0.25,
    EvaluationCategory.NEGOTIATION_RESOLUTION: 0.25,
}

# Minimum number of agent utterances for a meaningful evaluation
MIN_AGENT_UTTERANCES = 4

EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator of debt collection agent conversations.
Analyze the following transcript and evaluate the agent's performance across four categories.

Score each category from 0 to 100:
1. call_opening (20% weight): How well the agent opened the call - professional greeting, identification, stating purpose, verifying debtor identity.
2. compliance (30% weight): Adherence to debt collection regulations - Mini-Miranda disclosure, no harassment, proper tone, accurate information.
3. empathy_communication (25% weight): Use of empathetic language, active listening, acknowledgment of debtor's situation, clear communication.
4. negotiation_resolution (25% weight): Effectiveness in negotiating payment arrangements, offering solutions, handling objections, achieving resolution.

Also identify:
- Between 1 and 5 strengths demonstrated by the agent, each referencing a category and citing a specific transcript excerpt.
- Between 1 and 5 weaknesses observed, each referencing a category and citing a specific transcript excerpt.

Respond ONLY with valid JSON in this exact format:
{
  "category_scores": {
    "call_opening": <int 0-100>,
    "compliance": <int 0-100>,
    "empathy_communication": <int 0-100>,
    "negotiation_resolution": <int 0-100>
  },
  "strengths": [
    {
      "description": "<what the agent did well>",
      "category": "<call_opening|compliance|empathy_communication|negotiation_resolution>",
      "transcript_excerpt": "<exact quote from transcript>"
    }
  ],
  "weaknesses": [
    {
      "description": "<what the agent did poorly>",
      "category": "<call_opening|compliance|empathy_communication|negotiation_resolution>",
      "transcript_excerpt": "<exact quote from transcript>"
    }
  ]
}"""


class EvaluationEngine:
    """Analyzes completed transcripts and produces weighted competency scores.

    The evaluation engine calculates an overall weighted score from individual
    category scores and determines whether a session is too short for
    meaningful evaluation.
    """

    def __init__(self, llm_service: LLMServiceProtocol | None = None):
        """Initialize the EvaluationEngine.

        Args:
            llm_service: An LLM service instance for generating evaluations.
                If None, the evaluate() method will not be available.
        """
        self._llm_service = llm_service

    async def evaluate(
        self, session_id: UUID, transcript: list[dict], db=None
    ) -> EvaluationResult:
        """Run the full evaluation pipeline on a completed session transcript.

        Checks if the session is too short, and if not, calls the LLM to
        produce category scores, strengths, and weaknesses, then calculates
        the overall weighted score and persists the result.

        Args:
            session_id: The UUID of the session being evaluated.
            transcript: List of transcript entry dicts with 'speaker' and 'text' keys.
            db: Optional database session for persistence.

        Returns:
            EvaluationResult with scores, strengths, weaknesses, and is_too_short flag.

        Raises:
            ValueError: If no LLM service is configured.
        """
        # Check if session is too short
        if self.is_session_too_short(transcript):
            result = EvaluationResult(
                session_id=session_id,
                category_scores=[],
                overall_score=0.0,
                strengths=[StrengthItem(
                    description="Session too short for evaluation",
                    category=EvaluationCategory.CALL_OPENING,
                    transcript_excerpt="N/A",
                )],
                weaknesses=[WeaknessItem(
                    description="Session too short for evaluation",
                    category=EvaluationCategory.CALL_OPENING,
                    transcript_excerpt="N/A",
                )],
                is_too_short=True,
            )
            if db is not None:
                await self._persist_evaluation(session_id, result, db)
            return result

        if self._llm_service is None:
            raise ValueError("LLM service is required for evaluation")

        # Build transcript text for the prompt
        transcript_text = self._format_transcript(transcript)

        # Call LLM for evaluation
        messages = [
            LLMMessage(role="system", content=EVALUATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"Transcript:\n{transcript_text}"),
        ]

        llm_response = await self._llm_service.chat_completion(
            messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        # Parse LLM response
        parsed = json.loads(llm_response.content)

        # Build CompetencyScore objects
        category_scores_dict: dict[EvaluationCategory, int] = {}
        competency_scores: list[CompetencyScore] = []

        for category in EvaluationCategory:
            score_value = int(parsed["category_scores"][category.value])
            # Clamp to [0, 100]
            score_value = max(0, min(100, score_value))
            category_scores_dict[category] = score_value

            # Collect per-category strengths and weaknesses
            cat_strengths = [
                StrengthItem(
                    description=s["description"],
                    category=EvaluationCategory(s["category"]),
                    transcript_excerpt=s["transcript_excerpt"],
                )
                for s in parsed.get("strengths", [])
                if s.get("category") == category.value
            ]
            cat_weaknesses = [
                WeaknessItem(
                    description=w["description"],
                    category=EvaluationCategory(w["category"]),
                    transcript_excerpt=w["transcript_excerpt"],
                )
                for w in parsed.get("weaknesses", [])
                if w.get("category") == category.value
            ]

            competency_scores.append(
                CompetencyScore(
                    category=category,
                    score=score_value,
                    strengths=cat_strengths,
                    weaknesses=cat_weaknesses,
                )
            )

        # Calculate overall weighted score
        overall_score = self.calculate_overall_score(category_scores_dict)

        # Build top-level strengths and weaknesses (1-5 each)
        all_strengths = [
            StrengthItem(
                description=s["description"],
                category=EvaluationCategory(s["category"]),
                transcript_excerpt=s["transcript_excerpt"],
            )
            for s in parsed.get("strengths", [])
        ][:5]

        all_weaknesses = [
            WeaknessItem(
                description=w["description"],
                category=EvaluationCategory(w["category"]),
                transcript_excerpt=w["transcript_excerpt"],
            )
            for w in parsed.get("weaknesses", [])
        ][:5]

        # Ensure at least 1 strength and 1 weakness
        if not all_strengths:
            all_strengths = [StrengthItem(
                description="Agent participated in the call",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt=transcript[0].get("text", "N/A") if transcript else "N/A",
            )]
        if not all_weaknesses:
            all_weaknesses = [WeaknessItem(
                description="No specific weaknesses identified",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt=transcript[0].get("text", "N/A") if transcript else "N/A",
            )]

        result = EvaluationResult(
            session_id=session_id,
            category_scores=competency_scores,
            overall_score=overall_score,
            strengths=all_strengths,
            weaknesses=all_weaknesses,
            is_too_short=False,
        )

        # Persist with retry
        if db is not None:
            await self._persist_evaluation(session_id, result, db)

        return result

    def _format_transcript(self, transcript: list[dict]) -> str:
        """Format transcript entries into readable text for the LLM prompt.

        Args:
            transcript: List of transcript entry dicts.

        Returns:
            Formatted transcript string.
        """
        lines = []
        for entry in transcript:
            speaker = entry.get("speaker", "unknown").upper()
            text = entry.get("text", "")
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    async def _persist_evaluation(
        self, session_id: UUID, result: EvaluationResult, db
    ) -> None:
        """Persist the evaluation result to the database with retry logic.

        Args:
            session_id: The session UUID.
            result: The EvaluationResult to persist.
            db: The database session.
        """
        from app.models import Evaluation

        async def _do_persist():
            evaluation = Evaluation(
                session_id=session_id,
                overall_score=result.overall_score,
                category_scores=[cs.model_dump() for cs in result.category_scores],
                strengths=[s.model_dump() for s in result.strengths],
                weaknesses=[w.model_dump() for w in result.weaknesses],
                is_too_short=result.is_too_short,
            )
            db.add(evaluation)
            await db.commit()

        await retry_db_operation(
            _do_persist,
            session_id=str(session_id),
            data=result.model_dump(),
        )

    def calculate_overall_score(self, scores: dict[EvaluationCategory, int]) -> float:
        """Calculate the overall weighted score from category scores.

        Applies the defined category weights:
        - Call Opening: 20%
        - Compliance: 30%
        - Empathy & Communication: 25%
        - Negotiation & Resolution: 25%

        Args:
            scores: Dictionary mapping each EvaluationCategory to an integer
                score in [0, 100].

        Returns:
            The weighted overall score as a float in [0, 100].

        Raises:
            ValueError: If any score is outside [0, 100] or if required
                categories are missing.
        """
        # Validate all required categories are present
        for category in EvaluationCategory:
            if category not in scores:
                raise ValueError(
                    f"Missing score for category '{category.value}'. "
                    f"All four categories are required."
                )

        # Validate each score is in [0, 100]
        for category, score in scores.items():
            if not (0 <= score <= 100):
                raise ValueError(
                    f"Score for '{category.value}' must be between 0 and 100, "
                    f"got {score}."
                )

        # Calculate weighted sum
        overall = sum(
            scores[category] * weight
            for category, weight in CATEGORY_WEIGHTS.items()
        )

        return overall

    def is_session_too_short(self, transcript: list[dict]) -> bool:
        """Determine if a session transcript is too short for evaluation.

        A session is considered too short if the agent has fewer than 4
        utterances in the transcript.

        Args:
            transcript: List of transcript entry dicts, each containing at
                minimum a 'speaker' field with value "agent" or "debtor".

        Returns:
            True if the agent utterance count is less than 4, False otherwise.
        """
        agent_utterance_count = sum(
            1 for entry in transcript if entry.get("speaker") == "agent"
        )
        return agent_utterance_count < MIN_AGENT_UTTERANCES
