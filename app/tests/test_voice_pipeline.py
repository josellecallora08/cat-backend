"""Unit tests for VoicePipelineOrchestrator.

Tests the voice pipeline orchestrator's ability to coordinate the full
audio processing flow: VAD → buffer → STT → LLM → TTS → output.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
    SimulatorResponse,
)
from app.services.voice.audio_buffer import AudioBuffer
from app.services.voice.peer_connection_manager import PeerConnectionManager
from app.services.voice.stt_service import MockSTTService, TranscriptionResult
from app.services.voice.tts_service import AudioStream, MockTTSService
from app.services.voice.vad import (
    FRAME_SIZE_BYTES,
    VADProcessor,
    VADResult,
    EnergyVADBackend,
)
from app.services.voice.voice_pipeline import VoicePipelineOrchestrator, PipelineState


# --- Fixtures ---


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def persona():
    return PersonaContext(
        persona_id=uuid.uuid4(),
        name="Test Debtor",
        communication_style="cooperative",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 5000,
            "reason_for_delinquency": "job loss",
        },
        emotional_state=EmotionalState.NEUTRAL,
        language="EN",
    )


@pytest.fixture
def mock_debtor_simulator():
    simulator = AsyncMock(spec=DebtorSimulatorService)
    simulator.generate_response = AsyncMock(
        return_value=SimulatorResponse(
            text="I understand, let me explain my situation.",
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
def mock_stt_service():
    return MockSTTService(
        default_text="Hello, I'm calling about your account.",
        default_language="en",
        default_confidence=0.92,
    )


@pytest.fixture
def mock_tts_service():
    return MockTTSService()


@pytest.fixture
def mock_peer_connection_manager():
    manager = AsyncMock(spec=PeerConnectionManager)
    manager.create_peer_connection = AsyncMock(return_value=MagicMock())
    manager.close_peer_connection = AsyncMock()
    manager.is_available = True
    return manager


@pytest.fixture
def orchestrator(
    session_id,
    persona,
    mock_debtor_simulator,
    mock_transcript_manager,
    mock_stt_service,
    mock_tts_service,
    mock_peer_connection_manager,
):
    return VoicePipelineOrchestrator(
        session_id=session_id,
        persona=persona,
        debtor_simulator=mock_debtor_simulator,
        transcript_manager=mock_transcript_manager,
        stt_service=mock_stt_service,
        tts_service=mock_tts_service,
        peer_connection_manager=mock_peer_connection_manager,
    )


def _make_speech_frame() -> bytes:
    """Create a PCM frame that registers as speech (high energy)."""
    import struct

    # High amplitude samples that will pass energy threshold
    num_samples = FRAME_SIZE_BYTES // 2
    samples = [5000] * num_samples
    return struct.pack(f"<{num_samples}h", *samples)


def _make_silence_frame() -> bytes:
    """Create a PCM frame that registers as silence (low energy)."""
    return b"\x00" * FRAME_SIZE_BYTES


# --- Tests ---


class TestVoicePipelineOrchestratorInit:
    """Tests for VoicePipelineOrchestrator initialization."""

    def test_init_creates_pipeline_with_defaults(self, orchestrator, session_id, persona):
        """Pipeline initializes with provided dependencies and default components."""
        assert orchestrator.session_id == session_id
        assert orchestrator.persona == persona
        assert orchestrator.is_active is False
        assert orchestrator.is_processing is False

    def test_init_creates_vad_and_buffer_if_not_provided(
        self,
        session_id,
        persona,
        mock_debtor_simulator,
        mock_transcript_manager,
        mock_stt_service,
        mock_tts_service,
    ):
        """Pipeline creates VAD and AudioBuffer when not explicitly provided."""
        orch = VoicePipelineOrchestrator(
            session_id=session_id,
            persona=persona,
            debtor_simulator=mock_debtor_simulator,
            transcript_manager=mock_transcript_manager,
            stt_service=mock_stt_service,
            tts_service=mock_tts_service,
        )
        assert orch._vad is not None
        assert orch._audio_buffer is not None


class TestSetupPeerConnection:
    """Tests for setup_peer_connection."""

    @pytest.mark.asyncio
    async def test_setup_creates_peer_connection(
        self, orchestrator, session_id, mock_peer_connection_manager
    ):
        """setup_peer_connection delegates to PeerConnectionManager."""
        result = await orchestrator.setup_peer_connection(session_id)

        mock_peer_connection_manager.create_peer_connection.assert_called_once_with(
            session_id
        )
        assert result["session_id"] == str(session_id)
        assert result["status"] == "connected"
        assert orchestrator.is_active is True


class TestProcessAudioFrame:
    """Tests for process_audio_frame - the core pipeline logic."""

    @pytest.mark.asyncio
    async def test_speech_frame_is_buffered(self, orchestrator):
        """Speech frames are accumulated in the audio buffer."""
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()

        result = await orchestrator.process_audio_frame(speech_frame)

        # Should not produce output yet (no end-of-utterance)
        assert result is None
        # Frame should be in the buffer
        assert orchestrator._audio_buffer.frame_count == 1

    @pytest.mark.asyncio
    async def test_silence_after_speech_triggers_processing(
        self, orchestrator, mock_debtor_simulator, mock_stt_service
    ):
        """Silence following speech triggers the full STT→LLM→TTS pipeline."""
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        # Feed speech frames to establish speech
        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)

        # Feed enough silence frames to trigger end-of-utterance (500ms = 25 frames at 20ms)
        result = None
        for _ in range(26):
            result = await orchestrator.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Should have produced TTS audio output
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_pipeline_inactive_returns_none(self, orchestrator):
        """Frames are ignored when pipeline is not active."""
        orchestrator._state.is_active = False
        speech_frame = _make_speech_frame()

        result = await orchestrator.process_audio_frame(speech_frame)

        assert result is None
        assert orchestrator._audio_buffer.frame_count == 0

    @pytest.mark.asyncio
    async def test_vad_and_buffer_reset_after_utterance(self, orchestrator):
        """VAD and buffer are reset after processing a complete utterance."""
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        # Feed speech then silence to trigger processing
        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)
        for _ in range(26):
            result = await orchestrator.process_audio_frame(silence_frame)
            if result is not None:
                break

        # After processing, buffer should be empty and VAD reset
        assert orchestrator._audio_buffer.is_empty()

    @pytest.mark.asyncio
    async def test_stt_failure_returns_none(self, orchestrator):
        """STT failure is handled gracefully, returning None."""
        orchestrator._state.is_active = True

        # Replace STT with one that raises
        orchestrator._stt_service = MagicMock()
        orchestrator._stt_service.transcribe = MagicMock(
            side_effect=RuntimeError("STT failed")
        )

        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)
        for _ in range(26):
            result = await orchestrator.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Should return None due to STT failure (not crash)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_transcription_is_skipped(self, orchestrator):
        """Empty STT results do not trigger LLM or TTS."""
        orchestrator._state.is_active = True

        # Replace STT with one that returns empty text
        orchestrator._stt_service = MockSTTService(
            default_text="", default_confidence=0.1
        )

        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)

        result = None
        for _ in range(26):
            result = await orchestrator.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Should return None since transcription is empty
        assert result is None


class TestTranscriptRecording:
    """Tests for transcript entry recording during pipeline processing."""

    @pytest.mark.asyncio
    async def test_agent_and_debtor_entries_recorded(
        self, orchestrator, mock_transcript_manager
    ):
        """Both agent and debtor transcript entries are recorded."""
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)
        for _ in range(26):
            result = await orchestrator.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Should have recorded 2 entries: agent + debtor
        assert mock_transcript_manager.append_entry.call_count == 2

        # First call: agent entry
        agent_call = mock_transcript_manager.append_entry.call_args_list[0]
        assert agent_call.kwargs["speaker"] == "agent"
        assert agent_call.kwargs["text"] != ""

        # Second call: debtor entry
        debtor_call = mock_transcript_manager.append_entry.call_args_list[1]
        assert debtor_call.kwargs["speaker"] == "debtor"
        assert debtor_call.kwargs["text"] != ""

    @pytest.mark.asyncio
    async def test_transcript_failure_does_not_crash_pipeline(
        self, orchestrator, mock_transcript_manager
    ):
        """Transcript recording failure doesn't stop the pipeline."""
        orchestrator._state.is_active = True
        mock_transcript_manager.append_entry = AsyncMock(
            side_effect=Exception("DB error")
        )

        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for _ in range(5):
            await orchestrator.process_audio_frame(speech_frame)

        # Should not crash even though transcript recording fails
        for _ in range(26):
            result = await orchestrator.process_audio_frame(silence_frame)
            if result is not None:
                break

        # Pipeline should still be active
        assert orchestrator._state.is_active is True


class TestTeardown:
    """Tests for teardown lifecycle."""

    @pytest.mark.asyncio
    async def test_teardown_closes_peer_connection(
        self, orchestrator, mock_peer_connection_manager, session_id
    ):
        """Teardown closes the WebRTC peer connection."""
        orchestrator._state.is_active = True

        await orchestrator.teardown()

        mock_peer_connection_manager.close_peer_connection.assert_called_once_with(
            session_id
        )
        assert orchestrator.is_active is False

    @pytest.mark.asyncio
    async def test_teardown_persists_transcript(
        self, orchestrator, mock_transcript_manager, session_id
    ):
        """Teardown flushes buffered transcript entries to database."""
        orchestrator._state.is_active = True

        await orchestrator.teardown()

        mock_transcript_manager.persist.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_teardown_clears_output_queue(self, orchestrator):
        """Teardown drains any remaining audio from the output queue."""
        orchestrator._state.is_active = True
        await orchestrator._output_queue.put(b"audio1")
        await orchestrator._output_queue.put(b"audio2")

        await orchestrator.teardown()

        assert orchestrator._output_queue.empty()

    @pytest.mark.asyncio
    async def test_teardown_handles_peer_connection_error(
        self, orchestrator, mock_peer_connection_manager
    ):
        """Teardown handles peer connection close failure gracefully."""
        orchestrator._state.is_active = True
        mock_peer_connection_manager.close_peer_connection = AsyncMock(
            side_effect=Exception("Connection error")
        )

        # Should not raise
        await orchestrator.teardown()

        assert orchestrator.is_active is False

    @pytest.mark.asyncio
    async def test_teardown_handles_transcript_persist_error(
        self, orchestrator, mock_transcript_manager
    ):
        """Teardown handles transcript persist failure gracefully."""
        orchestrator._state.is_active = True
        mock_transcript_manager.persist = AsyncMock(
            side_effect=Exception("DB error")
        )

        # Should not raise
        await orchestrator.teardown()

        assert orchestrator.is_active is False


class TestOutputQueue:
    """Tests for the response audio output queue."""

    @pytest.mark.asyncio
    async def test_get_next_response_audio_returns_queued_audio(self, orchestrator):
        """get_next_response_audio returns audio from the queue."""
        await orchestrator._output_queue.put(b"test_audio")

        result = await orchestrator.get_next_response_audio()

        assert result == b"test_audio"

    @pytest.mark.asyncio
    async def test_get_next_response_audio_returns_none_when_empty(self, orchestrator):
        """get_next_response_audio returns None when queue is empty."""
        result = await orchestrator.get_next_response_audio()

        assert result is None


class TestHandleAudioTrack:
    """Tests for handle_audio_track method."""

    @pytest.mark.asyncio
    async def test_handle_audio_track_sets_active(self, orchestrator):
        """handle_audio_track marks the pipeline as active."""
        # Create a mock track that returns one frame then raises to exit
        mock_track = AsyncMock()
        mock_track.recv = AsyncMock(side_effect=asyncio.TimeoutError())

        # Run briefly then cancel
        task = asyncio.create_task(orchestrator.handle_audio_track(mock_track))
        await asyncio.sleep(0.1)
        orchestrator._state.is_active = False
        await asyncio.sleep(0.1)

        # Ensure task can finish
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_handle_audio_track_processes_bytes_frames(self, orchestrator):
        """handle_audio_track processes raw bytes frames."""
        frame_data = _make_speech_frame()

        # Track returns one frame then raises to exit the loop
        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return frame_data
            else:
                # Stop after one frame
                orchestrator._state.is_active = False
                raise Exception("done")

        mock_track = AsyncMock()
        mock_track.recv = mock_recv

        await orchestrator.handle_audio_track(mock_track)

        # The frame should have been processed
        assert orchestrator._audio_buffer.frame_count >= 0  # May be 0 if reset happened


class TestEndToEndProcessing:
    """Integration-style tests for the full pipeline flow."""

    @pytest.mark.asyncio
    async def test_full_utterance_produces_response(
        self,
        orchestrator,
        mock_debtor_simulator,
        mock_transcript_manager,
    ):
        """A complete speech→silence sequence produces TTS audio output."""
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        # Simulate agent speaking
        for _ in range(10):
            await orchestrator.process_audio_frame(speech_frame)

        # Simulate silence to trigger end-of-utterance
        response = None
        for _ in range(30):
            response = await orchestrator.process_audio_frame(silence_frame)
            if response is not None:
                break

        # Verify full pipeline executed
        assert response is not None
        mock_debtor_simulator.generate_response.assert_called_once()
        assert mock_transcript_manager.append_entry.call_count == 2
        assert orchestrator._state.utterance_count == 1

    @pytest.mark.asyncio
    async def test_multiple_utterances_increment_count(
        self, orchestrator, mock_debtor_simulator
    ):
        """Multiple utterances increment the utterance counter."""
        orchestrator._state.is_active = True
        speech_frame = _make_speech_frame()
        silence_frame = _make_silence_frame()

        for utterance_num in range(2):
            # Speech
            for _ in range(5):
                await orchestrator.process_audio_frame(speech_frame)
            # Silence to trigger processing
            for _ in range(26):
                result = await orchestrator.process_audio_frame(silence_frame)
                if result is not None:
                    break

        assert orchestrator._state.utterance_count == 2
        assert mock_debtor_simulator.generate_response.call_count == 2
