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


# Keywords used to interpret free-text `state_change` descriptions from
# script-defined Emotional_State_Rules into a directional step. Per design.md,
# `EmotionalStateRule.state_change` is free text written by Script authors
# (e.g. "increase_defensiveness", "decrease_anxiety", "increase_cooperation")
# rather than a structured numeric delta, so it must be interpreted heuristically.
_NEGATIVE_EMOTION_KEYWORDS = frozenset([
    "defensive",
    "defensiveness",
    "anxiety",
    "anxious",
    "hostility",
    "hostile",
    "anger",
    "angry",
    "distress",
    "fear",
    "resentment",
    "frustration",
    "irritation",
    "agitation",
])

_POSITIVE_EMOTION_KEYWORDS = frozenset([
    "cooperation",
    "cooperative",
    "trust",
    "calm",
    "relief",
    "receptiveness",
    "receptive",
    "comfort",
    "reassurance",
    "ease",
])


def _resolve_state_change_direction(state_change: str) -> int:
    """Resolve a free-text `state_change` description into a directional step.

    Documented heuristic (state_change is free text, not a structured numeric
    delta — this mapping is an interpretive design decision, not fully
    specified by the schema):

    1. Identify the emotion word's inherent polarity by substring match:
       - "Negative" emotion words (defensiveness, anxiety, hostility, anger,
         distress, fear, resentment, frustration, irritation, agitation)
         have polarity -1 (worse for the debtor's cooperativeness — e.g. more
         defensive/anxious debtors move toward HOSTILE).
       - "Positive" emotion words (cooperation, trust, calm, relief,
         receptiveness, comfort, reassurance, ease) have polarity +1 (better
         for cooperativeness — move toward COOPERATIVE).
       - Negative keywords are checked first, so a phrase mentioning both
         (unlikely in practice) resolves to the negative polarity.
    2. Identify the modifier:
       - "increase"/"more"/"heighten" keeps the polarity as-is (e.g.
         "increase_defensiveness" -> polarity -1 stays -1).
       - "decrease"/"less"/"reduce"/"lower" inverts the polarity (e.g.
         "decrease_anxiety" -> polarity -1 becomes +1, since less anxiety is
         an improvement).
       - Absence of either modifier keeps the polarity as-is (an author
         writing plain "cooperation" or "hostility" is describing the
         resulting state directly, not a rate of change).
    3. If no recognized emotion word is found at all, the direction is
       ambiguous and treated as no change (0) — the safest fallback since we
       cannot infer intent from unrecognized free text.

    Args:
        state_change: The free-text `state_change` description from a
            script-defined `EmotionalStateRule`.

    Returns:
        -1 (move toward HOSTILE), +1 (move toward COOPERATIVE), or 0 (no change).
    """
    text = state_change.lower()

    polarity = 0
    for kw in _NEGATIVE_EMOTION_KEYWORDS:
        if kw in text:
            polarity = -1
            break
    if polarity == 0:
        for kw in _POSITIVE_EMOTION_KEYWORDS:
            if kw in text:
                polarity = 1
                break

    if polarity == 0:
        return 0

    if "decrease" in text or "less" in text or "reduce" in text or "lower" in text:
        return -polarity
    return polarity


def _apply_directional_step(current: EmotionalState, direction: int) -> EmotionalState:
    """Apply a -1/0/+1 directional step to an emotional state, clamped to bounds."""
    if direction > 0:
        return EmotionalState(min(current.value + 1, EmotionalState.COOPERATIVE.value))
    elif direction < 0:
        return EmotionalState(max(current.value - 1, EmotionalState.HOSTILE.value))
    return current


def transition_emotional_state(
    current: EmotionalState,
    tone: AgentTone,
    emotional_state_rules: list[dict[str, Any]] | None = None,
    event: str | None = None,
) -> EmotionalState:
    """Transition the debtor's emotional state based on the agent's tone.

    When a script defines `emotional_state_rules`, this function first checks
    for a rule whose `trigger` matches (case-insensitive) either the classified
    `tone.value` (e.g. "aggressive", "empathetic", "neutral") or an explicit
    `event` string, and applies that rule's `state_change` (interpreted via
    `_resolve_state_change_direction`) instead of the hardcoded table (Req 4.5).
    When no rule matches, or `emotional_state_rules` is None/empty, this falls
    back to the existing hardcoded +1/-1/no-change behavior, unchanged.

    Hardcoded fallback rules:
    - EMPATHETIC tone: move +1 toward cooperative (higher value), clamped at COOPERATIVE(5)
    - AGGRESSIVE tone: move -1 toward hostile (lower value), clamped at HOSTILE(1)
    - NEUTRAL tone: no change

    Args:
        current: The current emotional state of the debtor.
        tone: The classified tone of the agent's message.
        emotional_state_rules: Optional list of script-defined rule dicts,
            each with a `trigger` and `state_change` string (mirrors the
            `EmotionalStateRule` schema). Defaults to None, which preserves
            100% backward-compatible behavior for callers with no script.
        event: Optional explicit event name to match against a rule's
            `trigger`, in addition to the classified tone.

    Returns:
        The new emotional state after applying the transition.
    """
    if emotional_state_rules:
        candidates = {tone.value.lower()}
        if event:
            candidates.add(event.strip().lower())

        for rule in emotional_state_rules:
            trigger = str(rule.get("trigger", "")).strip().lower()
            if trigger and trigger in candidates:
                direction = _resolve_state_change_direction(str(rule.get("state_change", "")))
                return _apply_directional_step(current, direction)

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


def select_opening_response(script_content: dict[str, Any] | None) -> str | None:
    """Select the first debtor utterance from a loaded Script_Version's content.

    When a script is present, the debtor's opening line is taken verbatim from
    the script contract's ``opening_response`` field instead of generating one
    via an LLM call (Req 4.4). This is a pure, side-effect-free extraction.

    Args:
        script_content: The validated ``ScriptContract`` content dict attached
            to the session/persona, or None if no script is loaded for this
            session (callers should fall back to the existing LLM-based flow).

    Returns:
        The verbatim ``opening_response`` string if a script is present,
        otherwise None.
    """
    if script_content is None:
        return None
    return script_content["opening_response"]


def match_trigger_phrase(
    agent_message: str, trigger_phrases: list[dict[str, Any]] | None
) -> str | None:
    """Scan `agent_message` for a script-defined trigger phrase and return its behavior.

    Before calling the LLM, `generate_response` scans the agent's message for
    any `trigger_phrases[*].phrase` from the loaded Script_Version's content
    (case-insensitive substring match) and, if found, applies the associated
    `behavior` (Req 4.6). This is a pure, side-effect-free lookup step -- no
    DB access, no LLM calls, no persona mutation.

    Tie-breaking rule: if `agent_message` contains more than one entry's
    `phrase` as a substring, the behavior of the FIRST matching entry in
    `trigger_phrases` list order is returned (list order, not phrase length
    or specificity, determines precedence).

    Args:
        agent_message: The agent's latest message text.
        trigger_phrases: Optional list of script-defined trigger phrase dicts,
            each with a `phrase` and `behavior` string (mirrors the
            `TriggerPhraseEntry` schema). Defaults to None, which means no
            script is loaded / no trigger phrases are defined.

    Returns:
        The `behavior` string of the first matching entry, or None if
        `trigger_phrases` is None/empty or no phrase matches.
    """
    if not trigger_phrases:
        return None

    lower_message = agent_message.lower()
    for entry in trigger_phrases:
        phrase = str(entry.get("phrase", ""))
        if phrase and phrase.lower() in lower_message:
            return entry.get("behavior")

    return None


def evaluate_escalation_conditions(
    agent_message: str, escalation_conditions: list[dict[str, Any]] | None
) -> tuple[str, bool] | None:
    """Evaluate the running conversation state against script-defined escalation conditions.

    Per design.md, Escalation_Conditions are "evaluated each turn against the
    running conversation state; when met, the corresponding `behavior` is
    applied, and if `ends_call=True` the call is terminated the same way the
    existing `[END_CALL]` marker works today" (Req 4.7). design.md does not
    specify the exact mechanism for evaluating "conversation state" against a
    free-text `condition` field, so this function makes an explicit,
    documented interpretive choice: it mirrors `match_trigger_phrase`'s
    approach and treats `condition` as a case-insensitive substring to match
    against `agent_message` -- the most directly available piece of
    conversation state at the point this would be called from
    `generate_response`. This keeps evaluation of the two structurally
    similar free-text fields (`trigger_phrases[*].phrase` and
    `escalation_conditions[*].condition`) consistent with each other.

    Tie-breaking rule: identical to `match_trigger_phrase` -- if
    `agent_message` matches more than one entry's `condition`, the
    `(behavior, ends_call)` of the FIRST matching entry in `escalation_conditions`
    list order is returned (list order determines precedence, not condition
    length or specificity).

    This function is pure and side-effect-free: it does not itself end the
    call or mutate any state. Wiring the returned `ends_call` flag into the
    actual call-termination logic (analogous to the existing `[END_CALL]`
    marker handling) is left to a later integration step in
    `generate_response`.

    Args:
        agent_message: The agent's latest message text.
        escalation_conditions: Optional list of script-defined escalation
            condition dicts, each with a `condition`, `behavior`, and
            `ends_call` field (mirrors the `EscalationConditionEntry` schema).
            Defaults to None, which means no script is loaded / no escalation
            conditions are defined.

    Returns:
        A `(behavior, ends_call)` tuple for the first matching entry, or None
        if `escalation_conditions` is None/empty or no condition matches.
    """
    if not escalation_conditions:
        return None

    lower_message = agent_message.lower()
    for entry in escalation_conditions:
        condition = str(entry.get("condition", ""))
        if condition and condition.lower() in lower_message:
            return entry.get("behavior"), bool(entry.get("ends_call", False))

    return None


def contains_prohibited_response(
    debtor_response_text: str, prohibited_responses: list[str] | None
) -> bool:
    """Check whether a generated debtor response matches a prohibited entry.

    Per design.md's `Prohibited_Response` glossary entry, `prohibited_responses`
    entries represent "a response category or content pattern" the debtor must
    never say -- free text, not exact strings to equality-match. This function
    makes the same explicit, documented interpretive choice as
    `match_trigger_phrase`/`evaluate_escalation_conditions` (Req 4.6, 4.7): it
    treats each `prohibited_responses` entry as a case-insensitive substring to
    match against `debtor_response_text`, keeping evaluation of this
    structurally similar free-text field consistent with the rest of this
    module's script-driven matching.

    This function is pure and side-effect-free: it does not itself trigger a
    regeneration retry or select a fallback line. Wiring the result into the
    actual retry/fallback behavior is handled by `generate_response` (Req 4.8).

    Args:
        debtor_response_text: The raw debtor response text generated by the
            LLM for this turn.
        prohibited_responses: Optional list of script-defined prohibited
            response strings (mirrors the `ScriptContract.prohibited_responses`
            field). Defaults to None, which means no script is loaded / no
            prohibited responses are defined.

    Returns:
        True if `debtor_response_text` contains any `prohibited_responses`
        entry as a case-insensitive substring, False otherwise (including
        when `prohibited_responses` is None/empty).
    """
    if not prohibited_responses:
        return False

    lower_response = debtor_response_text.lower()
    for entry in prohibited_responses:
        entry_text = str(entry).strip()
        if entry_text and entry_text.lower() in lower_response:
            return True

    return False


# Safe, generic, in-character but harmless debtor line returned by
# `generate_response` when every regeneration attempt still matches a
# `prohibited_responses` entry (Req 4.8). Short and content-neutral so it
# never itself risks matching a prohibited pattern.
SAFE_DEFAULT_DEBTOR_RESPONSE: str = "I'm not sure how to respond to that right now."


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

    # Maximum number of LLM regeneration attempts (the initial call plus this
    # many retries) made when a generated response matches a
    # `prohibited_responses` entry, before falling back to
    # `SAFE_DEFAULT_DEBTOR_RESPONSE` (Req 4.8). Kept small and bounded so a
    # persistently-matching LLM can never loop indefinitely or delay the
    # Agent's turn unreasonably: 1 initial attempt + 2 retries = 3 total
    # LLM calls at most for a single `generate_response` invocation.
    _MAX_PROHIBITED_RESPONSE_RETRIES: int = 2

    async def generate_response(
        self,
        persona: PersonaContext,
        agent_message: str,
        script_content: dict[str, Any] | None = None,
    ) -> SimulatorResponse:
        """Generate a debtor response to the agent's message.

        Classifies the agent's tone, transitions the emotional state,
        detects the agent's language, builds a conversation prompt, and
        calls the LLM to generate an in-character response.

        When `script_content` is provided and defines `prohibited_responses`,
        the generated response is checked against them (Req 4.8): on a
        match, the LLM call is retried up to `_MAX_PROHIBITED_RESPONSE_RETRIES`
        additional times, and if every attempt still matches a prohibited
        response, `SAFE_DEFAULT_DEBTOR_RESPONSE` is returned instead so
        prohibited content never reaches the Agent. When `script_content` is
        None (the default), behavior is unchanged from before this check was
        added -- no prohibited-response checking and no extra LLM calls.

        Args:
            persona: The current persona context with conversation history.
            agent_message: The agent's latest message text.
            script_content: Optional loaded `ScriptContract` content dict for
                the pinned `Script_Version`, used to read
                `prohibited_responses`. Defaults to None, which preserves
                existing behavior for callers with no script loaded.

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

        prohibited_responses = (
            script_content.get("prohibited_responses") if script_content is not None else None
        )

        # 6. Call LLM, with a bounded regeneration retry if the response
        # matches a prohibited pattern (Req 4.8). When no script/prohibited
        # responses are defined, this loop always exits after the first
        # attempt, matching prior behavior exactly.
        debtor_response_text = ""
        for attempt in range(self._MAX_PROHIBITED_RESPONSE_RETRIES + 1):
            response = await self.llm_service.chat_completion(
                messages,
                temperature=0.8,
            )
            debtor_response_text = response.content.strip()

            if not contains_prohibited_response(debtor_response_text, prohibited_responses):
                break
        else:
            # Every attempt (initial + all retries) matched a prohibited
            # response -- fall back to the safe default line so prohibited
            # content never reaches the Agent.
            debtor_response_text = SAFE_DEFAULT_DEBTOR_RESPONSE

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

IMPORTANT CONTEXT:
- You are receiving an INCOMING call from an UNKNOWN number. You do NOT know who is calling.
- You have NO idea this is a collection call until the agent properly introduces themselves, states their company name, and explains the purpose of the call.
- When you first answer, respond as any normal person would to an unknown caller — ask who they are, what they want, why they're calling.
- Do NOT mention debt, payments, loans, or money UNTIL the agent has clearly explained they are calling about a specific debt.
- Only after the agent identifies themselves and their purpose should you respond based on your personality traits and financial situation.
- If the agent fails to identify themselves properly, keep asking "Sino po kayo?" or "Anong kailangan ninyo?" — do not volunteer information.

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

        # Append global admin prompt if provided
        if global_prompt:
            return base_prompt + f"\n\nADMIN GLOBAL INSTRUCTIONS:\n{global_prompt}"
        return base_prompt


def match_payment_condition(
    agent_message: str, payment_conditions: list[dict[str, Any]] | None
) -> tuple[str, bool] | None:
    """Match `agent_message` against script-defined payment conditions.

    Per design.md, "when agent input matches a `payment_conditions[*].condition`,
    the debtor's reply is steered by the `accepted`/`term` fields" (Req 4.9).
    design.md does not specify the exact mechanism for matching free-text
    `condition` against agent input, so this function makes the same explicit,
    documented interpretive choice as `match_trigger_phrase` and
    `evaluate_escalation_conditions`: it treats each entry's `condition` as a
    case-insensitive substring to match against `agent_message`, keeping
    evaluation of this structurally similar free-text field consistent with
    the rest of this module's script-driven matching.

    Tie-breaking rule: identical to `match_trigger_phrase` and
    `evaluate_escalation_conditions` -- if `agent_message` matches more than
    one entry's `condition`, the `(term, accepted)` of the FIRST matching
    entry in `payment_conditions` list order is returned (list order
    determines precedence, not condition length or specificity).

    This function is pure and side-effect-free: it does not itself steer a
    debtor reply or call the LLM. Wiring the returned `(term, accepted)` into
    `generate_response`'s conversation flow is left to a later integration
    step, mirroring how `match_trigger_phrase` and
    `evaluate_escalation_conditions` were left unwired at their own
    introduction.

    Args:
        agent_message: The agent's latest message text.
        payment_conditions: Optional list of script-defined payment condition
            dicts, each with a `condition`, `term`, and `accepted` field
            (mirrors the `PaymentConditionEntry` schema). Defaults to None,
            which means no script is loaded / no payment conditions are
            defined.

    Returns:
        A `(term, accepted)` tuple for the first matching entry, or None if
        `payment_conditions` is None/empty or no condition matches.
    """
    if not payment_conditions:
        return None

    lower_message = agent_message.lower()
    for entry in payment_conditions:
        condition = str(entry.get("condition", ""))
        if condition and condition.lower() in lower_message:
            return entry.get("term"), bool(entry.get("accepted", False))

    return None

def evaluate_conversation_goal_completion(
    agent_message: str, conversation_goal: dict[str, Any] | None
) -> str | None:
    """Evaluate whether the script-defined conversation goal is satisfied.

    Per design.md's "Conversation goal" bullet, "`completion_condition` is
    evaluated each turn; when satisfied, the call ends per `target_outcome`"
    (Req 4.10). Unlike `trigger_phrases`/`escalation_conditions`/
    `payment_conditions`, `ConversationGoal` is a single object (not a list of
    entries) with a single free-text `completion_condition` field and a
    single free-text `target_outcome` field -- design.md does not specify a
    more granular matching mechanism, nor a structurally distinct "input" to
    evaluate `completion_condition` against beyond "conversation state" in
    general. This function makes the same explicit, documented interpretive
    choice as `match_trigger_phrase`, `evaluate_escalation_conditions`, and
    `match_payment_condition`, extended to this single-field case: it treats
    `completion_condition` as a case-insensitive substring to match against
    `agent_message` -- the most directly available piece of conversation
    state at the point this would be called from `generate_response` --
    keeping evaluation of this free-text field consistent with the rest of
    this module's script-driven matching.

    This function is pure and side-effect-free: it does not itself end the
    call. Wiring the returned `target_outcome` into the actual
    call-termination logic is left to a later integration step in
    `generate_response`, mirroring how `match_trigger_phrase`,
    `evaluate_escalation_conditions`, and `match_payment_condition` were left
    unwired at their own introduction.

    Args:
        agent_message: The agent's latest message text.
        conversation_goal: Optional script-defined conversation goal dict with
            `target_outcome` and `completion_condition` string fields (mirrors
            the `ConversationGoal` schema). Defaults to None, which means no
            script is loaded / no conversation goal is defined.

    Returns:
        The `target_outcome` string if `completion_condition` matches
        (case-insensitive substring) `agent_message`, or None if
        `conversation_goal` is None, `completion_condition` is missing/empty,
        or no match is found.
    """
    if not conversation_goal:
        return None

    completion_condition = str(conversation_goal.get("completion_condition", ""))
    if not completion_condition:
        return None

    if completion_condition.lower() in agent_message.lower():
        return conversation_goal.get("target_outcome")

    return None
