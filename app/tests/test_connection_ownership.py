"""Regression tests for releasing DB connections before slow external work."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.scenarios import GenerateScenarioRequest, generate_scenario
from app.api.sessions import ConversationMessage, _active_personas, _send_message_locked
from app.api.voice import _load_voice_session_context
from app.services.debtor_simulator import EmotionalState, PersonaContext, SimulatorResponse


@pytest.mark.asyncio
async def test_voice_context_releases_initial_read_transaction():
    session_id = uuid.uuid4()
    script_version_id = uuid.uuid4()
    db = MagicMock()
    db.get = AsyncMock(
        return_value=SimpleNamespace(
            script_version_id=script_version_id,
            persona_context={"name": "Maria"},
        )
    )
    db.rollback = AsyncMock()

    with patch(
        "app.api.voice.load_script_content",
        new=AsyncMock(return_value={"opening_response": "Hello"}),
    ):
        persona, script = await _load_voice_session_context(db, session_id)

    assert persona == {"name": "Maria"}
    assert script == {"opening_response": "Hello"}
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_text_message_releases_transaction_before_llm():
    session_id = uuid.uuid4()
    db = MagicMock()
    db.rollback = AsyncMock()
    session = SimpleNamespace(
        status="active",
        script_version_id=None,
        persona_context={},
    )
    transcript_manager = MagicMock()
    transcript_manager.append_entry = AsyncMock()
    transcript_manager.persist = AsyncMock()
    simulator = MagicMock()

    async def _generate_response(*args, **kwargs):
        assert db.rollback.await_count == 1
        return SimulatorResponse(
            text="We can discuss it.",
            emotional_state=EmotionalState.NEUTRAL,
            language="EN",
        )

    simulator.generate_response = AsyncMock(side_effect=_generate_response)
    _active_personas[session_id] = PersonaContext(
        persona_id=uuid.uuid4(),
        name="Maria",
        communication_style="cooperative",
        financial_circumstances={},
        emotional_state=EmotionalState.NEUTRAL,
        language="EN",
    )

    try:
        with (
            patch("app.api.sessions.get_session_service", new=AsyncMock(return_value=session)),
            patch("app.api.sessions.load_script_content", new=AsyncMock(return_value=None)),
            patch("app.api.sessions.LLMService", return_value=MagicMock()),
            patch("app.services.debtor_simulator.DebtorSimulatorService", return_value=simulator),
            patch(
                "app.services.transcript_manager.TranscriptManager", return_value=transcript_manager
            ),
        ):
            response = await _send_message_locked(
                session_id,
                ConversationMessage(text="Can you pay today?"),
                db,
            )
    finally:
        _active_personas.pop(session_id, None)

    assert response.text == "We can discuss it."
    db.rollback.assert_awaited_once()
    transcript_manager.persist.assert_awaited_once_with(session_id)


@pytest.mark.asyncio
async def test_scenario_generation_releases_auth_transaction_before_llm():
    db = MagicMock()
    db.rollback = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    llm = MagicMock()

    async def _chat_completion(*args, **kwargs):
        db.rollback.assert_awaited_once()
        return SimpleNamespace(
            content=(
                '{"name":"Payment Plan","scenario_type":"FINANCIAL_HARDSHIP",'
                '"description":"A realistic scenario.","debtor_profile":'
                '{"name":"Maria Santos","outstanding_balance":"5000.00"}}'
            )
        )

    llm.chat_completion = AsyncMock(side_effect=_chat_completion)
    admin = SimpleNamespace(id=uuid.uuid4(), role="admin")

    with (
        patch("app.api.scenarios.LLMService", return_value=llm),
        patch("app.api.scenarios.event_broadcaster.emit", new=AsyncMock()),
    ):
        result = await generate_scenario(
            GenerateScenarioRequest(prompt="Create a hardship scenario"),
            db=db,
            admin=admin,
        )

    assert result.name == "Payment Plan"
    db.rollback.assert_awaited_once()
    db.commit.assert_awaited_once()
