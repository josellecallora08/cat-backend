"""End-to-end wiring tests for the voice pipeline.

Verifies the complete pipeline chain:
WebRTC audio → VAD → AudioBuffer → STT → DebtorSimulator → TTS → output queue

Also verifies that transcript entries are recorded in real-time for both
agent and debtor utterances.

Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.3
"""

import asyncio
import struct
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
    SimulatorResponse,
)
from app.services.voice.audio_buffer import AudioBuffer
from app.services.voice.peer_connection_manager import PeerConnectionManager
from app.services.voice.pipeline_factory import create_voice_pipeline
from app.services.voice.stt_service import MockSTTService, TranscriptionResult
from app.services.voice.tts_service import AudioStream, MockTTSService
from app.services.voice.vad import FRAME_SIZE_BYTES, VADProcessor
from app.services.voice.voice_pipeline import VoicePipelineOrchestrator


# --- Helpers ---


def _make_speech_frame() -> bytes:
    """Create a PCM frame that registers as speech (high energy)."""
    num_samples = FRAME_SIZE_BYTES // 2
    samples = [5000] * num_samples
    return struct.pack(f"<{num_samples}h", *samples)


def _make_silence_frame() -> bytes:
    """Create a PCM frame that registers as silence (low energy)."""
    return b"\x00" * FRAME_SIZE_BYTES


# --- Fixtures ---


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def persona():
    return PersonaContext(
        persona_id=uuid.uuid4(),
        name="Maria Santos",
        communication_style="anxious",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 12000,
            "reason_for_delinquency": "medical emergency",
        },
        emotional_state=EmotionalState.DEFENSIVE,
        language="EN",
    )


@pytest.fixture
def mock_stt_service():
    return MockSTTService(
        default_text="I'd like to discuss your payment options.",
        default_language="en",
        default_confidence=0.95,
    )


@pytest.fixture
def mock_tts_service():
    return MockTTSService()


@pytest.fixture
def mock_debtor_simulator():
    simulator = AsyncMock(spec=DebtorSimulatorService)
    simulator.generate_response = AsyncMock(
        return_value=SimulatorResponse(
            text="I've been having trouble with my finances. Can we work something out?",
            emotional_state=EmotionalState.RECEPTIVE,
            language="EN",
        )
    )
    return simulator


@pytest.fixture
def mock_transcript_manager():
    manager = AsyncMock()
    manager.append_entry = AsyncMock()
    manager.persist = AsyncMock()
    return manager


@pytest.fixture
def mock_peer_connection_manager():
    manager = AsyncMock(spec=PeerConnectionManager)
    manager.create_peer_connection = AsyncMock(return_value=MagicMock())
    manager.close_peer_connection = AsyncMock()
    manager.is_available = True
    return manager


@pytest.fixture
def pipeline(
    session_id,
    persona,
    mock_debtor_simulator,
    mock_transcript_manager,
    mock_stt_service,
    mock_tts_service,
    mock_peer_connection_manager,
):
    """Create a VoicePipelineOrchestrator with all mocked components."""
    return VoicePipelineOrchestrator(
        session_id=session_id,
        persona=persona,
        debtor_simulator=mock_debtor_simulator,
        transcript_manager=mock_transcript_manager,
        stt_service=mock_stt_service,
        tts_service=mock_tts_service,
        peer_connection_manager=mock_peer_connection_manager,
    )


# --- End-to-End Wiring Tests ---


class TestPipelineEndToEndWiring:
    """Verify the complete pipeline chain is wired correctly."""

    @pytest.mark.asyncio
    async def test_full_chain_webrtc_to_output(
        self, pipeline, mock_stt_service, mock_debtor_simulator, mock_tts_service
    ):
        """Audio frames flow through: VAD → buffer → STT → DebtorSimulator → TTS → output.

        Validates the complete wiring from audio input to audio output.
        Requirements: 3.1, 3.2, 3.3, 3.4
        """
        pipeline._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        # 1. Feed speech frames (simulating agent talking via WebRTC)
        for _ in range(10):
            result = await pipeline.process_audio_frame(speech_frame)
            assert result is None  # No output yet during speech

        # 2. Feed silence frames to trigger end-of-utterance (VAD detects 500ms silence)
        response_audio = None
        for _ in range(30):
            response_audio = await pipeline.process_audio_frame(silence_frame)
            if response_audio is not None:
                break

        # 3. Verify complete chain executed
        assert response_audio is not None, "Pipeline should produce TTS audio output"
        assert len(response_audio) > 0, "Audio output should have content"

        # Verify STT was called with buffered audio
        assert mock_stt_service.call_count == 1
        assert mock_stt_service.last_audio is not None
        assert len(mock_stt_service.last_audio) > 0

        # Verify DebtorSimulator received the transcribed text
        mock_debtor_simulator.generate_response.assert_called_once()
        call_args = mock_debtor_simulator.generate_response.call_args
        assert call_args[0][1] == "I'd like to discuss your payment options."

        # Verify TTS was called with debtor response
        assert len(mock_tts_service.synthesize_calls) == 1
        synthesized_text, language = mock_tts_service.synthesize_calls[0]
        assert "trouble with my finances" in synthesized_text

        # Verify output was queued
        queued_audio = await pipeline.get_next_response_audio()
        # The output queue already has the audio (it was put there internally)
        # response_audio was returned directly from process_audio_frame

    @pytest.mark.asyncio
    async def test_transcript_records_agent_and_debtor_in_realtime(
        self, pipeline, mock_transcript_manager, session_id
    ):
        """Both agent and debtor utterances are recorded to TranscriptManager.

        Validates real-time transcript recording as required by Requirements 4.1, 4.3.
        """
        pipeline._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        # Process a complete utterance
        for _ in range(5):
            await pipeline.process_audio_frame(speech_frame)
        for _ in range(26):
            result = await pipeline.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Verify both transcript entries recorded
        assert mock_transcript_manager.append_entry.call_count == 2

        # Verify agent entry
        agent_call = mock_transcript_manager.append_entry.call_args_list[0]
        assert agent_call.kwargs["session_id"] == session_id
        assert agent_call.kwargs["speaker"] == "agent"
        assert agent_call.kwargs["text"] == "I'd like to discuss your payment options."
        assert isinstance(agent_call.kwargs["timestamp"], datetime)

        # Verify debtor entry
        debtor_call = mock_transcript_manager.append_entry.call_args_list[1]
        assert debtor_call.kwargs["session_id"] == session_id
        assert debtor_call.kwargs["speaker"] == "debtor"
        assert "trouble with my finances" in debtor_call.kwargs["text"]
        assert isinstance(debtor_call.kwargs["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_transcript_timestamps_are_chronological(
        self, pipeline, mock_transcript_manager
    ):
        """Agent timestamp precedes debtor timestamp in each exchange.

        Validates: Requirement 4.3 (timestamp with millisecond precision).
        """
        pipeline._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await pipeline.process_audio_frame(speech_frame)
        for _ in range(26):
            result = await pipeline.process_audio_frame(silence_frame)
            if result is not None:
                break

        agent_call = mock_transcript_manager.append_entry.call_args_list[0]
        debtor_call = mock_transcript_manager.append_entry.call_args_list[1]

        agent_ts = agent_call.kwargs["timestamp"]
        debtor_ts = debtor_call.kwargs["timestamp"]

        # Agent utterance is recorded before debtor response
        assert agent_ts <= debtor_ts

    @pytest.mark.asyncio
    async def test_multiple_exchanges_all_recorded(
        self, pipeline, mock_transcript_manager, mock_debtor_simulator
    ):
        """Multiple conversation turns are all recorded sequentially.

        Validates continuous real-time recording (Requirement 4.1).
        """
        pipeline._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        # Simulate 3 complete exchanges
        for _ in range(3):
            for _ in range(5):
                await pipeline.process_audio_frame(speech_frame)
            for _ in range(26):
                result = await pipeline.process_audio_frame(silence_frame)
                if result is not None:
                    break

        # 3 exchanges × 2 entries (agent + debtor) = 6 transcript entries
        assert mock_transcript_manager.append_entry.call_count == 6

        # Verify alternating agent/debtor pattern
        speakers = [
            call.kwargs["speaker"]
            for call in mock_transcript_manager.append_entry.call_args_list
        ]
        assert speakers == ["agent", "debtor", "agent", "debtor", "agent", "debtor"]

    @pytest.mark.asyncio
    async def test_vad_correctly_gates_stt(self, pipeline, mock_stt_service):
        """VAD prevents STT from being called on silence-only input.

        Validates: Requirement 3.1 (STT only receives actual speech).
        """
        pipeline._state.is_active = True
        silence_frame = _make_silence_frame()

        # Feed only silence frames
        for _ in range(50):
            result = await pipeline.process_audio_frame(silence_frame)
            assert result is None

        # STT should never have been called
        assert mock_stt_service.call_count == 0

    @pytest.mark.asyncio
    async def test_audio_buffer_accumulates_speech_frames(self, pipeline):
        """AudioBuffer accumulates frames during speech for STT.

        Validates the buffering stage between VAD and STT.
        """
        pipeline._state.is_active = True
        speech_frame = _make_speech_frame()

        # Feed 10 speech frames
        for _ in range(10):
            await pipeline.process_audio_frame(speech_frame)

        # Buffer should have accumulated 10 frames
        assert pipeline._audio_buffer.frame_count == 10
        assert pipeline._audio_buffer.duration_ms == 200  # 10 × 20ms

    @pytest.mark.asyncio
    async def test_output_queue_receives_tts_audio(self, pipeline):
        """TTS output is placed in the output queue for WebRTC delivery.

        Validates: Requirement 3.2 (TTS output to WebRTC).
        """
        pipeline._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await pipeline.process_audio_frame(speech_frame)
        for _ in range(26):
            result = await pipeline.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Output queue should have the response audio
        queued = await pipeline.get_next_response_audio()
        assert queued is not None
        assert len(queued) > 0

    @pytest.mark.asyncio
    async def test_teardown_persists_transcript(
        self, pipeline, mock_transcript_manager, session_id
    ):
        """Pipeline teardown flushes transcript to persistent storage.

        Validates: Requirement 4.1 (transcript persistence).
        """
        pipeline._state.is_active = True

        await pipeline.teardown()

        mock_transcript_manager.persist.assert_called_once_with(session_id)
        assert pipeline.is_active is False


class TestPipelineFactory:
    """Tests for the pipeline_factory.create_voice_pipeline function."""

    def test_create_voice_pipeline_with_mock_services(self, session_id, persona):
        """Factory creates a fully configured pipeline with provided services."""
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_stt = MockSTTService()
        mock_tts = MockTTSService()
        mock_pcm = AsyncMock(spec=PeerConnectionManager)
        mock_pcm.is_available = True

        orchestrator = create_voice_pipeline(
            session_id=session_id,
            persona=persona,
            db=mock_db,
            llm_service=mock_llm,
            stt_service=mock_stt,
            tts_service=mock_tts,
            peer_connection_manager=mock_pcm,
        )

        assert isinstance(orchestrator, VoicePipelineOrchestrator)
        assert orchestrator.session_id == session_id
        assert orchestrator.persona == persona
        assert orchestrator._stt_service is mock_stt
        assert orchestrator._tts_service is mock_tts
        assert orchestrator._peer_connection_manager is mock_pcm

    def test_factory_creates_default_vad_with_500ms_threshold(self, session_id, persona):
        """Factory creates VAD with 500ms silence threshold by default."""
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_stt = MockSTTService()
        mock_tts = MockTTSService()

        orchestrator = create_voice_pipeline(
            session_id=session_id,
            persona=persona,
            db=mock_db,
            llm_service=mock_llm,
            stt_service=mock_stt,
            tts_service=mock_tts,
        )

        assert orchestrator._vad.silence_threshold_ms == 500

    def test_factory_creates_default_buffer_with_30s_max(self, session_id, persona):
        """Factory creates audio buffer with 30s max duration by default."""
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_stt = MockSTTService()
        mock_tts = MockTTSService()

        orchestrator = create_voice_pipeline(
            session_id=session_id,
            persona=persona,
            db=mock_db,
            llm_service=mock_llm,
            stt_service=mock_stt,
            tts_service=mock_tts,
        )

        assert orchestrator._audio_buffer.max_duration_ms == 30_000

    def test_factory_accepts_custom_vad_and_buffer(self, session_id, persona):
        """Factory uses provided VAD and buffer overrides."""
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_stt = MockSTTService()
        mock_tts = MockTTSService()

        custom_vad = VADProcessor(silence_threshold_ms=300)
        custom_buffer = AudioBuffer(max_duration_ms=15_000)

        orchestrator = create_voice_pipeline(
            session_id=session_id,
            persona=persona,
            db=mock_db,
            llm_service=mock_llm,
            stt_service=mock_stt,
            tts_service=mock_tts,
            vad=custom_vad,
            audio_buffer=custom_buffer,
        )

        assert orchestrator._vad.silence_threshold_ms == 300
        assert orchestrator._audio_buffer.max_duration_ms == 15_000

    @pytest.mark.asyncio
    async def test_factory_pipeline_processes_audio_end_to_end(
        self, session_id, persona
    ):
        """Pipeline created by factory correctly processes audio through full chain."""
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_stt = MockSTTService(
            default_text="Hello, this is agent calling.",
            default_confidence=0.9,
        )
        mock_tts = MockTTSService()

        # Create a mock debtor simulator that will be injected
        orchestrator = create_voice_pipeline(
            session_id=session_id,
            persona=persona,
            db=mock_db,
            llm_service=mock_llm,
            stt_service=mock_stt,
            tts_service=mock_tts,
        )

        # Override the debtor simulator with a mock for testing
        mock_debtor_sim = AsyncMock()
        mock_debtor_sim.generate_response = AsyncMock(
            return_value=SimulatorResponse(
                text="Hi, what can I help you with?",
                emotional_state=EmotionalState.NEUTRAL,
                language="EN",
            )
        )
        orchestrator._debtor_simulator = mock_debtor_sim

        # Process audio through the factory-created pipeline
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)
        response = None
        for _ in range(26):
            response = await orchestrator.process_audio_frame(silence_frame)
            if response is not None:
                break

        assert response is not None
        assert len(response) > 0
        mock_debtor_sim.generate_response.assert_called_once()
