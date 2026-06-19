"""Debtor Simulator service for generating personas and managing conversations."""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from app.services.llm_service import LLMMessage, LLMServiceProtocol


class EmotionalState(IntEnum):
    """Emotional state scale from hostile to cooperative."""

    HOSTILE = 1
    DEFENSIVE = 2
    NEUTRAL = 3
    RECEPTIVE = 4
    COOPERATIVE = 5


class AgentTone(Enum):
    """Classification of the agent's tone in a message."""

    EMPATHETIC = "empathetic"
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"


# Keywords used for tone classification
_EMPATHETIC_KEYWORDS = frozenset([
    "understand",
    "sorry",
    "help",
    "appreciate",
    "concern",
    "difficult",
    "together",
    "support",
    "listen",
    "flexible",
    "option",
    "work with you",
    "i hear you",
    "take your time",
    "no pressure",
    "how can i help",
    "i understand",
    "empathize",
    "care",
    "comfortable",
])

_AGGRESSIVE_KEYWORDS = frozenset([
    "must",
    "demand",
    "threat",
    "pay now",
    "immediately",
    "consequence",
    "legal action",
    "final notice",
    "no choice",
    "forced",
    "garnish",
    "lawsuit",
    "failure to pay",
    "collection agency",
    "report",
    "deadline",
    "overdue",
    "unacceptable",
    "refuse",
    "penalties",
])


def classify_agent_tone(message: str) -> AgentTone:
    """Classify the agent's tone based on keyword analysis.

    Uses keyword-based classification to determine whether the agent's message
    is empathetic, aggressive, or neutral.

    Args:
        message: The agent's message text.

    Returns:
        AgentTone indicating the classified tone.
    """
    lower_message = message.lower()

    empathetic_count = sum(1 for kw in _EMPATHETIC_KEYWORDS if kw in lower_message)
    aggressive_count = sum(1 for kw in _AGGRESSIVE_KEYWORDS if kw in lower_message)

    if empathetic_count > aggressive_count:
        return AgentTone.EMPATHETIC
    elif aggressive_count > empathetic_count:
        return AgentTone.AGGRESSIVE
    else:
        return AgentTone.NEUTRAL


def transition_emotional_state(current: EmotionalState, tone: AgentTone) -> EmotionalState:
    """Transition the debtor's emotional state based on the agent's tone.

    Rules:
    - EMPATHETIC tone: move +1 toward cooperative (higher value), clamped at COOPERATIVE(5)
    - AGGRESSIVE tone: move -1 toward hostile (lower value), clamped at HOSTILE(1)
    - NEUTRAL tone: no change

    Args:
        current: The current emotional state of the debtor.
        tone: The classified tone of the agent's message.

    Returns:
        The new emotional state after applying the transition.
    """
    if tone == AgentTone.EMPATHETIC:
        new_value = min(current.value + 1, EmotionalState.COOPERATIVE.value)
        return EmotionalState(new_value)
    elif tone == AgentTone.AGGRESSIVE:
        new_value = max(current.value - 1, EmotionalState.HOSTILE.value)
        return EmotionalState(new_value)
    else:
        return current


@dataclass
class Message:
    """A single conversation message."""

    role: str  # "agent" or "debtor"
    content: str


@dataclass
class SimulatorResponse:
    """Response from the debtor simulator after processing an agent message."""

    text: str
    emotional_state: EmotionalState
    language: str  # EN, TL, TAGLISH


@dataclass
class PersonaContext:
    """Generated debtor persona with all contextual information."""

    persona_id: uuid.UUID
    name: str
    communication_style: str  # cooperative, evasive, hostile, anxious
    financial_circumstances: dict[str, Any]  # income_level, debt_amount, reason_for_delinquency
    emotional_state: EmotionalState
    conversation_history: list[Message] = field(default_factory=list)
    language: str = "EN"  # EN, TL, TAGLISH


# Valid communication styles for persona generation
VALID_COMMUNICATION_STYLES = {"cooperative", "evasive", "hostile", "anxious"}

# Default emotional state mapping from scenario personality profiles
PERSONALITY_TO_INITIAL_STATE: dict[str, EmotionalState] = {
    "cooperative": EmotionalState.COOPERATIVE,
    "anxious": EmotionalState.DEFENSIVE,
    "hostile": EmotionalState.HOSTILE,
    "evasive": EmotionalState.NEUTRAL,
}

# Common Tagalog words used for language detection
_TAGALOG_WORDS = frozenset([
    "ako", "ikaw", "siya", "kami", "tayo", "sila", "ang", "ng", "sa",
    "na", "po", "opo", "hindi", "oo", "wala", "meron", "bakit", "paano",
    "kailan", "saan", "sino", "ano", "mga", "naman", "lang", "din", "rin",
    "ba", "kasi", "talaga", "namin", "natin", "niyo", "nila", "ko", "mo",
    "niya", "ito", "iyan", "iyon", "dito", "diyan", "doon", "pera", "bayad",
    "utang", "trabaho", "pamilya", "salamat", "pasensya", "kuya", "ate",
    "magbayad", "problema", "tulong", "mahirap", "bayaran", "kailangan",
])


def detect_language(text: str) -> str:
    """Detect the language of a message using simple heuristic.

    Checks for the presence of Tagalog words. If a mix of English and Tagalog
    is detected, classifies as TAGLISH. Pure Tagalog returns TL, else EN.

    Args:
        text: The message text to analyze.

    Returns:
        Language code: "EN", "TL", or "TAGLISH".
    """
    words = text.lower().split()
    if not words:
        return "EN"

    tagalog_count = sum(1 for w in words if w.strip(".,!?;:'\"") in _TAGALOG_WORDS)
    total_words = len(words)

    if total_words == 0:
        return "EN"

    tagalog_ratio = tagalog_count / total_words

    if tagalog_ratio >= 0.5:
        return "TL"
    elif tagalog_ratio >= 0.15:
        return "TAGLISH"
    else:
        return "EN"


def _build_persona_generation_prompt(scenario: dict[str, Any]) -> str:
    """Build a system prompt instructing the LLM to generate persona details.

    Args:
        scenario: Dictionary with scenario data including debtor_profile.

    Returns:
        System prompt string for LLM persona generation.
    """
    debtor_profile = scenario.get("debtor_profile", {})
    name = debtor_profile.get("name", "Unknown")
    personality = debtor_profile.get("personality_profile", "neutral")
    balance = debtor_profile.get("outstanding_balance", 0)
    days_past_due = debtor_profile.get("days_past_due", 0)
    conversation_goal = debtor_profile.get("conversation_goal", "resolve debt")
    scenario_type = scenario.get("scenario_type", "FINANCIAL_HARDSHIP")
    description = scenario.get("description", "")

    return f"""You are a persona generator for a debt collection training simulator.

Generate a realistic debtor persona based on the following scenario:
- Scenario type: {scenario_type}
- Scenario description: {description}
- Debtor name: {name}
- Personality profile: {personality}
- Outstanding balance: {balance}
- Days past due: {days_past_due}
- Conversation goal: {conversation_goal}

You MUST respond with ONLY a valid JSON object (no markdown, no extra text) with these exact fields:
{{
    "name": "<full name for the persona>",
    "communication_style": "<one of: cooperative, evasive, hostile, anxious>",
    "financial_circumstances": {{
        "income_level": "<low, medium, or high>",
        "debt_amount": <numeric amount>,
        "reason_for_delinquency": "<brief explanation>"
    }},
    "emotional_state": <integer 1-5 where 1=hostile, 2=defensive, 3=neutral, 4=receptive, 5=cooperative>,
    "language": "TAGLISH"
}}

The persona should be consistent with the personality profile ({personality}) and scenario type ({scenario_type}).
The communication_style MUST be one of: cooperative, evasive, hostile, anxious.
The emotional_state MUST be an integer from 1 to 5.
The persona is Filipino and speaks in Taglish (a mix of Tagalog and English)."""


def _parse_persona_response(response_content: str, scenario: dict[str, Any]) -> PersonaContext:
    """Parse the LLM response into a PersonaContext.

    Args:
        response_content: Raw LLM response text (expected JSON).
        scenario: Original scenario data for fallback values.

    Returns:
        PersonaContext with parsed persona details.

    Raises:
        ValueError: If the response cannot be parsed into a valid persona.
    """
    # Strip markdown code fences if present
    content = response_content.strip()
    if content.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = content.index("\n")
        content = content[first_newline + 1:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response is not valid JSON: {e}") from e

    # Validate and extract fields
    name = data.get("name", "").strip()
    if not name:
        raise ValueError("Persona name is empty")

    communication_style = data.get("communication_style", "").strip().lower()
    if communication_style not in VALID_COMMUNICATION_STYLES:
        raise ValueError(
            f"Invalid communication_style '{communication_style}'. "
            f"Must be one of: {VALID_COMMUNICATION_STYLES}"
        )

    financial_circumstances = data.get("financial_circumstances", {})
    if not isinstance(financial_circumstances, dict):
        raise ValueError("financial_circumstances must be a dictionary")
    if not financial_circumstances:
        raise ValueError("financial_circumstances cannot be empty")

    # Validate emotional state
    raw_state = data.get("emotional_state", 3)
    try:
        emotional_state = EmotionalState(int(raw_state))
    except (ValueError, TypeError):
        # Default to NEUTRAL if invalid
        emotional_state = EmotionalState.NEUTRAL

    language = data.get("language", "EN").upper()
    if language not in ("EN", "TL", "TAGLISH"):
        language = "EN"

    return PersonaContext(
        persona_id=uuid.uuid4(),
        name=name,
        communication_style=communication_style,
        financial_circumstances=financial_circumstances,
        emotional_state=emotional_state,
        language=language,
    )


class DebtorSimulatorService:
    """Service for generating debtor personas and managing simulated conversations.

    Uses an LLM service (Ollama/vLLM via OpenAI-compatible API) to generate
    realistic debtor personas based on training scenarios.
    """

    def __init__(self, llm_service: LLMServiceProtocol):
        """Initialize with an LLM service instance.

        Args:
            llm_service: An implementation of the LLM service protocol.
        """
        self.llm_service = llm_service

    async def generate_persona(self, scenario: dict[str, Any]) -> PersonaContext:
        """Generate a debtor persona from scenario data.

        Builds a system prompt from the scenario, calls the LLM to produce
        persona details, and parses the response into a PersonaContext.

        Args:
            scenario: Dictionary containing scenario data with keys:
                - debtor_profile: dict with name, outstanding_balance, days_past_due,
                  personality_profile, conversation_goal
                - scenario_type: str
                - description: str (optional)

        Returns:
            PersonaContext with generated persona details.

        Raises:
            ValueError: If LLM response cannot be parsed into valid persona.
            httpx.HTTPStatusError: If LLM API returns error.
            httpx.TimeoutException: If LLM API times out.
        """
        system_prompt = _build_persona_generation_prompt(scenario)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content="Generate the debtor persona as specified. Respond with only the JSON object.",
            ),
        ]

        response = await self.llm_service.chat_completion(
            messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        return _parse_persona_response(response.content, scenario)

    async def generate_response(
        self, persona: PersonaContext, agent_message: str
    ) -> SimulatorResponse:
        """Generate a debtor response to the agent's message.

        Classifies the agent's tone, transitions the emotional state,
        detects the agent's language, builds a conversation prompt, and
        calls the LLM to generate an in-character response.

        Args:
            persona: The current persona context with conversation history.
            agent_message: The agent's latest message text.

        Returns:
            SimulatorResponse with generated text, updated emotional state,
            and detected language.

        Raises:
            httpx.HTTPStatusError: If LLM API returns error.
            httpx.TimeoutException: If LLM API times out.
        """
        # 1. Classify agent tone
        tone = classify_agent_tone(agent_message)

        # 2. Transition emotional state
        new_state = transition_emotional_state(persona.emotional_state, tone)
        persona.emotional_state = new_state

        # 3. Detect language
        detected_language = detect_language(agent_message)
        persona.language = detected_language

        # 4. Build system prompt
        system_prompt = self._build_conversation_system_prompt(persona, detected_language)

        # 5. Build messages with conversation history
        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]

        for msg in persona.conversation_history:
            if msg.role == "agent":
                messages.append(LLMMessage(role="user", content=msg.content))
            else:
                messages.append(LLMMessage(role="assistant", content=msg.content))

        # Add the current agent message
        messages.append(LLMMessage(role="user", content=agent_message))

        # 6. Call LLM
        response = await self.llm_service.chat_completion(
            messages,
            temperature=0.8,
        )

        debtor_response_text = response.content.strip()

        # 7. Update conversation history
        persona.conversation_history.append(Message(role="agent", content=agent_message))
        persona.conversation_history.append(Message(role="debtor", content=debtor_response_text))

        return SimulatorResponse(
            text=debtor_response_text,
            emotional_state=new_state,
            language=detected_language,
        )

    def _build_conversation_system_prompt(
        self, persona: PersonaContext, language: str
    ) -> str:
        """Build the system prompt for conversation response generation.

        Args:
            persona: The persona context with traits and state.
            language: The detected language to respond in.

        Returns:
            System prompt string for the LLM.
        """
        emotional_state_name = persona.emotional_state.name.lower()

        language_instruction = {
            "EN": "You MUST respond in Taglish (a natural mix of Tagalog and English, as commonly spoken in the Philippines). Example: 'Hindi ko po kaya mag-pay ng full amount ngayon, pwede bang mag-request ng installment?'",
            "TL": "You MUST respond in Taglish (a natural mix of Tagalog and English, as commonly spoken in the Philippines). Example: 'Hindi ko po kaya mag-pay ng full amount ngayon, pwede bang mag-request ng installment?'",
            "TAGLISH": "You MUST respond in Taglish (a natural mix of Tagalog and English, as commonly spoken in the Philippines). Example: 'Hindi ko po kaya mag-pay ng full amount ngayon, pwede bang mag-request ng installment?'",
        }.get(language, "You MUST respond in Taglish (a natural mix of Tagalog and English).")

        financial_info = persona.financial_circumstances
        financial_summary = ", ".join(
            f"{k}: {v}" for k, v in financial_info.items()
        )

        return f"""You are roleplaying as {persona.name}, a debtor in a collection call simulation.

CHARACTER TRAITS:
- Communication style: {persona.communication_style}
- Current emotional state: {emotional_state_name}
- Financial circumstances: {financial_summary}

INSTRUCTIONS:
- Stay in character at all times. You are {persona.name}, not an AI assistant.
- Respond naturally as a person with the above traits and emotional state would.
- Your emotional state is {emotional_state_name}. Let this influence your tone and willingness to cooperate.
- Keep responses concise and conversational (1-3 sentences typically).
- {language_instruction}
- Do NOT break character or acknowledge that you are an AI.
- Do NOT use quotation marks around your response.
- If the agent is being overly aggressive, threatening, harassing, or you feel disrespected, you may end the call by including exactly "[END_CALL]" at the very end of your message — but ONLY do this as an absolute last resort when the agent's behavior is truly unacceptable (repeated threats, yelling, insults). Normal pressure or firm language is NOT enough to hang up. Do NOT write action markers like "*hangs up*" or "*ends call*" in your response text — just say your final words naturally and append [END_CALL].
- If the conversation reaches a natural conclusion (payment arranged, dispute resolved, etc.), you may also end politely.
- If the agent is rambling, repeating themselves, or saying something confusing, you may interrupt with a short interjection like "Teka lang po..." or "Wait, ano po yun?" — keep interruptions to 1-8 words only.
- Do NOT hang up just because the agent mentions the debt or asks for payment — that is expected in a collection call."""
