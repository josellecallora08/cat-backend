"""Factory for creating fully configured VoicePipelineOrchestrator instances.

Creates and wires all voice pipeline components for a session:
WebRTC audio → VAD → AudioBuffer → STT → DebtorSimulator → TTS → WebRTC output

Each pipeline instance records both agent and debtor utterances to the
TranscriptManager in real-time during the session.

Latency targets (measurable only with actual hardware/models):
- WebRTC one-way audio latency: <300ms (Requirement 3.4)
- End-to-end response time (end-of-utterance → debtor voice output): <2s (Requirement 3.3)
- TTS first-byte latency: <500ms (Requirement 3.2)

Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.3
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.debtor_simulator import DebtorSimulatorService, PersonaContext
from app.services.llm_service import LLMServiceProtocol
from app.services.transcript_manager import TranscriptManager
from app.services.voice.audio_buffer import AudioBuffer
from app.services.voice.peer_connection_manager import PeerConnectionManager
from app.services.voice.stt_service import STTService, STTServiceProtocol
from app.services.voice.tts_service import TTSService, TTSServiceProtocol
from app.services.voice.vad import VADProcessor
from app.services.voice.voice_pipeline import VoicePipelineOrchestrator

logger = logging.getLogger(__name__)


def create_voice_pipeline(
    session_id: UUID,
    persona: PersonaContext,
    db: AsyncSession,
    llm_service: LLMServiceProtocol,
    *,
    stt_service: STTServiceProtocol | None = None,
    tts_service: TTSServiceProtocol | None = None,
    peer_connection_manager: PeerConnectionManager | None = None,
    vad: VADProcessor | None = None,
    audio_buffer: AudioBuffer | None = None,
) -> VoicePipelineOrchestrator:
    """Create a fully configured VoicePipelineOrchestrator for a session.

    Wires the complete voice pipeline:
        WebRTC audio → VAD → AudioBuffer → STT → DebtorSimulator → TTS → output queue

    Transcript entries are recorded in real-time for both agent and debtor
    utterances via the TranscriptManager.

    Latency targets (hardware-dependent, documented for reference):
        - WebRTC one-way latency: <300ms
        - End-to-end response (utterance end → voice delivery): <2s
        - TTS first-byte: <500ms

    Args:
        session_id: The session this pipeline belongs to.
        persona: The debtor persona context for this session.
        db: Async database session for transcript persistence.
        llm_service: LLM service for debtor response generation.
        stt_service: Optional STT service override (defaults to STTService).
        tts_service: Optional TTS service override (defaults to TTSService).
        peer_connection_manager: Optional WebRTC manager override.
        vad: Optional VAD processor override (defaults to VADProcessor with 500ms threshold).
        audio_buffer: Optional audio buffer override (defaults to AudioBuffer with 30s max).

    Returns:
        A fully configured VoicePipelineOrchestrator ready to handle audio.

    Raises:
        RuntimeError: If STT or TTS services fail to initialize.
    """
    # Create TranscriptManager for real-time recording
    transcript_manager = TranscriptManager(db)

    # Create DebtorSimulator with LLM backend
    debtor_simulator = DebtorSimulatorService(llm_service)

    # Initialize STT (lazily loads model on first transcription)
    if stt_service is None:
        stt_service = STTService(model_size="medium", device="cpu", compute_type="float32")

    # Initialize TTS (may raise if piper is not available)
    if tts_service is None:
        try:
            tts_service = TTSService(voice_model="en_US-lessac-medium")
        except (ImportError, Exception) as e:
            logger.warning(
                "TTS service unavailable, pipeline will not produce voice output: %s", e
            )
            raise RuntimeError(f"TTS service initialization failed: {e}") from e

    # Initialize VAD with 500ms silence threshold for end-of-utterance detection
    if vad is None:
        vad = VADProcessor(silence_threshold_ms=500)

    # Initialize audio buffer with 30s max duration
    if audio_buffer is None:
        audio_buffer = AudioBuffer(max_duration_ms=30_000)

    # Initialize WebRTC peer connection manager
    if peer_connection_manager is None:
        peer_connection_manager = PeerConnectionManager()

    orchestrator = VoicePipelineOrchestrator(
        session_id=session_id,
        persona=persona,
        debtor_simulator=debtor_simulator,
        transcript_manager=transcript_manager,
        stt_service=stt_service,
        tts_service=tts_service,
        peer_connection_manager=peer_connection_manager,
        vad=vad,
        audio_buffer=audio_buffer,
    )

    logger.info(
        "Session %s: voice pipeline created (VAD threshold=%dms, buffer max=%dms)",
        session_id,
        vad.silence_threshold_ms,
        audio_buffer.max_duration_ms,
    )

    return orchestrator
