"""Coaching Engine service for identifying mistakes and generating recommendations.

Analyzes completed transcripts alongside evaluation results to identify
specific agent mistakes, explain why behaviors were ineffective, and
suggest recommended alternative responses.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

import json
import logging
from typing import Dict, List
from uuid import UUID

from app.schemas import (
    CoachingReportSchema,
    EvaluationCategory,
    EvaluationResult,
    MistakeItem,
)
from app.services.db_retry import retry_db_operation
from app.services.llm_service import LLMMessage, LLMServiceProtocol

logger = logging.getLogger(__name__)


COACHING_SYSTEM_PROMPT = """You are an expert coach for debt collection agents. \
You have been given a transcript of a conversation between an agent and a debtor, \
along with the evaluation results identifying weaknesses.

Your task is to identify specific mistakes the agent made during the conversation. \
For each mistake:
1. Reference the exact position in the transcript (0-based index of the utterance)
2. Quote the relevant excerpt from the transcript
3. Assign a category from: call_opening, compliance, empathy_communication, negotiation_resolution
4. Explain WHY this was ineffective in the context of debt collection best practices
5. Provide a recommended alternative response the agent could have used — written in TAGLISH (natural mix of Tagalog and English as spoken in Philippine call centers)

IMPORTANT for recommended_alternative:
- Write the alternative in Taglish, the way a professional Filipino collection agent would actually speak on a call.
- Example: "Magandang araw po, ako po si [Agent Name] from [Company]. Tumatawag po ako regarding sa outstanding balance ninyo na 18,000 pesos. Pwede po ba natin pag-usapan kung paano natin ma-resolve ito?"
- Do NOT write alternatives in pure formal English. Use the natural Taglish that agents use in real calls.
- Keep the professional tone but use the natural Filipino-English code-switching.

Focus on the weaknesses identified in the evaluation. Only identify genuine mistakes \
where the agent's response failed to meet professional standards.

Respond ONLY with valid JSON in this exact format:
{
  "mistakes": [
    {
      "transcript_position": <int, 0-based index>,
      "transcript_excerpt": "<exact quote from transcript>",
      "category": "<call_opening|compliance|empathy_communication|negotiation_resolution>",
      "explanation": "<why this was ineffective — write in English>",
      "recommended_alternative": "<what the agent should have said — write in TAGLISH>"
    }
  ]
}

If the agent made no mistakes and all criteria were met, respond with:
{
  "mistakes": []
}"""


class CoachingEngine:
    """Identifies specific mistakes and generates actionable improvement recommendations.

    The coaching engine uses evaluation weaknesses as a guide to pinpoint
    exact moments in the transcript where the agent could have performed
    better, providing explanations and alternative responses.
    """

    def __init__(self, llm_service: LLMServiceProtocol):
        """Initialize the CoachingEngine.

        Args:
            llm_service: An LLM service instance for generating coaching reports.
        """
        self._llm_service = llm_service

    async def generate_report(
        self,
        session_id: UUID,
        transcript: list[dict],
        evaluation: EvaluationResult,
        db=None,
    ) -> CoachingReportSchema:
        """Generate a coaching report for a completed session.

        Builds a prompt combining the transcript and evaluation weaknesses,
        calls the LLM to identify mistakes, parses the response into
        MistakeItem objects grouped by category, and optionally persists
        the report to the database.

        Args:
            session_id: The UUID of the session being coached.
            transcript: List of transcript entry dicts with 'speaker' and 'text' keys.
            evaluation: The EvaluationResult containing scores and weaknesses.
            db: Optional database session for persistence.

        Returns:
            CoachingReportSchema with mistakes grouped by category.
        """
        # Build user prompt with transcript + evaluation context
        user_prompt = self._build_user_prompt(transcript, evaluation)

        # Call LLM with JSON response format
        messages = [
            LLMMessage(role="system", content=COACHING_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        llm_response = await self._llm_service.chat_completion(
            messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        # Parse LLM response into MistakeItem objects
        parsed = json.loads(llm_response.content)
        raw_mistakes = parsed.get("mistakes", [])

        # Convert to MistakeItem objects
        mistake_items: List[MistakeItem] = []
        for raw in raw_mistakes:
            try:
                item = MistakeItem(
                    transcript_position=int(raw["transcript_position"]),
                    transcript_excerpt=raw["transcript_excerpt"],
                    category=EvaluationCategory(raw["category"]),
                    explanation=raw["explanation"],
                    recommended_alternative=raw["recommended_alternative"],
                )
                mistake_items.append(item)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed mistake item: %s (error: %s)", raw, e)
                continue

        # Group mistakes by category
        mistakes_by_category: Dict[EvaluationCategory, List[MistakeItem]] = {}
        for item in mistake_items:
            if item.category not in mistakes_by_category:
                mistakes_by_category[item.category] = []
            mistakes_by_category[item.category].append(item)

        # Determine if no mistakes were found
        total_mistakes = len(mistake_items)
        no_mistakes = total_mistakes == 0

        # Build the coaching report
        report = CoachingReportSchema(
            session_id=session_id,
            mistakes_by_category=mistakes_by_category,
            total_mistakes=total_mistakes,
            no_mistakes=no_mistakes,
        )

        # Persist with retry wrapper when db is provided
        if db is not None:
            await self._persist_report(session_id, report, db)

        return report

    def _build_user_prompt(
        self, transcript: list[dict], evaluation: EvaluationResult
    ) -> str:
        """Build the user prompt combining transcript and evaluation context.

        Args:
            transcript: List of transcript entry dicts.
            evaluation: The evaluation result with weaknesses.

        Returns:
            Formatted prompt string for the LLM.
        """
        # Format transcript
        transcript_lines = []
        for i, entry in enumerate(transcript):
            speaker = entry.get("speaker", "unknown").upper()
            text = entry.get("text", "")
            transcript_lines.append(f"[{i}] {speaker}: {text}")
        transcript_text = "\n".join(transcript_lines)

        # Format evaluation weaknesses
        weakness_lines = []
        for weakness in evaluation.weaknesses:
            weakness_lines.append(
                f"- Category: {weakness.category.value}, "
                f"Issue: {weakness.description}, "
                f"Excerpt: \"{weakness.transcript_excerpt}\""
            )
        weaknesses_text = "\n".join(weakness_lines) if weakness_lines else "None identified"

        # Format category scores
        score_lines = []
        for cs in evaluation.category_scores:
            score_lines.append(f"- {cs.category.value}: {cs.score}/100")
        scores_text = "\n".join(score_lines) if score_lines else "No scores available"

        return (
            f"## Transcript\n{transcript_text}\n\n"
            f"## Category Scores\n{scores_text}\n\n"
            f"## Identified Weaknesses\n{weaknesses_text}\n\n"
            f"Overall Score: {evaluation.overall_score}/100\n\n"
            f"Please identify specific mistakes in the transcript based on the weaknesses above."
        )

    async def _persist_report(
        self, session_id: UUID, report: CoachingReportSchema, db
    ) -> None:
        """Persist the coaching report to the database with retry logic.

        Args:
            session_id: The session UUID.
            report: The CoachingReportSchema to persist.
            db: The database session.
        """
        from app.models import CoachingReport

        async def _do_persist():
            # Serialize mistakes_by_category to JSON-compatible format
            serialized_mistakes = {
                category.value: [item.model_dump() for item in items]
                for category, items in report.mistakes_by_category.items()
            }

            coaching_report = CoachingReport(
                session_id=session_id,
                mistakes_by_category=serialized_mistakes,
                total_mistakes=report.total_mistakes,
                no_mistakes=report.no_mistakes,
            )
            db.add(coaching_report)
            await db.commit()

        await retry_db_operation(
            _do_persist,
            session_id=str(session_id),
            data=report.model_dump(),
        )
