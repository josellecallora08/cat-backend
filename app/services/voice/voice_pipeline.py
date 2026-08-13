"""Voice Pipeline Orchestrator managing per-session audio processing.

Coordinates the full voice pipeline for a single session:
WebRTC audio → VAD → AudioBuffer → STT → DebtorSimulator → TTS → output

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.services.debtor_simulator import DebtorSimulatorService, PersonaContext
from app.services.transcript_manager import TranscriptManager
from app.services.voice.audio_buffer import AudioBuffer
from app.services.voice.peer_connection_manager import PeerConnectionManager
from app.services.voice.stt_service import STTServiceProtocol
from app.services.voice.tts_service import TTSServiceProtocol
from app.services.voice.vad import VADProcessor


logger = logging.getLogger(__name__)


@dataclass
class CallEndSignal:
    """Sentinel value placed in the output queue to signal call termination."""

    reason: str
    target_outcome: str | None = None


@dataclass
class PipelineState:
    """Internal state tracking for the voice pipeline."""

    is_active: bool = False
    is_processing: bool = False
    utterance_count: int = 0


class VoicePipelineOrchestrator:
    """Manages a single session's voice pipeline lifecycle.

    Orchestrates the flow: incoming audio → VAD → buffer → STT → LLM → TTS → output.

    Each instance is tied to a single session and coordinates the peer connection,
    audio processing components, and transcript recording.
    """

    def __init__(
        self,
        session_id: UUID,
        persona: PersonaContext,
        debtor_simulator: DebtorSimulatorService,
        transcript_manager: TranscriptManager,
        stt_service: STTServiceProtocol,
        tts_service: TTSServiceProtocol,
        peer_connection_manager: PeerConnectionManager | None = None,
        vad: VADProcessor | None = None,
        audio_buffer: AudioBuffer | None = None,
        script_content: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the voice pipeline orchestrator.

        Args:
            session_id: The session this pipeline belongs to.
            persona: The debtor persona context for this session.
            debtor_simulator: Service for generating debtor responses.
            transcript_manager: Service for recording transcript entries.
            stt_service: Speech-to-text service.
            tts_service: Text-to-speech service.
            peer_connection_manager: WebRTC peer connection manager (optional).
            vad: Voice activity detection processor (created if not provided).
            audio_buffer: Audio frame buffer (created if not provided).
            script_content: Optional loaded ScriptContract content dict for
                the pinned ScriptVersion, used for script-driven behavior.
        """
        self.session_id = session_id
        self.persona = persona
        self._debtor_simulator = debtor_simulator
        self._transcript_manager = transcript_manager
        self._stt_service = stt_service
        self._tts_service = tts_service
        self._peer_connection_manager = peer_connection_manager or PeerConnectionManager()
        self._vad = vad or VADProcessor()
        self._audio_buffer = audio_buffer or AudioBuffer()
        self._script_content = script_content
        self._state = PipelineState()
        self._output_queue: asyncio.Queue[bytes | CallEndSignal] = asyncio.Queue()
        self._peer_connection: Any = None

    @property
    def is_active(self) -> bool:
        """Whether the pipeline is currently active."""
        return self._state.is_active

    @property
    def is_processing(self) -> bool:
        """Whether the pipeline is currently processing an utterance."""
        return self._state.is_processing

    async def setup_peer_connection(self, session_id: UUID) -> dict:
        """Configure audio codecs and create the WebRTC peer connection.

        Delegates to PeerConnectionManager to create the RTCPeerConnection
        and configure audio handling.

        Args:
            session_id: The session identifier for this connection.

        Returns:
            Dict with peer connection info or SDP details.

        Raises:
            RuntimeError: If aiortc is not available.
        """
        self._peer_connection = await self._peer_connection_manager.create_peer_connection(
            session_id
        )
        self._state.is_active = True
        logger.info("Session %s: peer connection setup complete", session_id)
        return {"session_id": str(session_id), "status": "connected"}

    async def handle_audio_track(self, track: Any) -> None:
        """Receive and process incoming audio frames from a WebRTC track.

        Starts an async loop that reads frames from the track and routes
        them through the audio processing pipeline.

        Args:
            track: A WebRTC MediaStreamTrack providing audio frames.
        """
        self._state.is_active = True
        logger.info("Session %s: handling audio track", self.session_id)

        try:
            while self._state.is_active:
                try:
                    frame = await asyncio.wait_for(
                        track.recv(), timeout=5.0
                    )
                    # Extract raw PCM bytes from the frame
                    if hasattr(frame, "to_ndarray"):
                        # aiortc AudioFrame - convert to bytes
                        import numpy as np

                        audio_array = frame.to_ndarray()
                        pcm_bytes = audio_array.astype(np.int16).tobytes()
                    elif isinstance(frame, bytes):
                        pcm_bytes = frame
                    else:
                        pcm_bytes = bytes(frame)

                    await self.process_audio_frame(pcm_bytes)
                except TimeoutError:
                    # No frame received within timeout - continue listening
                    continue
                except Exception as e:
                    if not self._state.is_active:
                        break
                    logger.error(
                        "Session %s: error receiving audio frame: %s",
                        self.session_id,
                        e,
                    )
                    break
        finally:
            logger.info("Session %s: audio track handling ended", self.session_id)

    async def process_audio_frame(self, frame: bytes) -> bytes | None:
        """Process a single audio frame through the VAD → buffer → STT → LLM → TTS pipeline.

        Routes the frame through VAD for speech detection, accumulates in the
        audio buffer during speech, and on end-of-utterance triggers the full
        processing pipeline.

        Args:
            frame: Raw PCM audio frame data (16kHz, 16-bit mono, 20ms).

        Returns:
            TTS audio bytes if a complete utterance was processed, None otherwise.
        """
        if not self._state.is_active:
            return None

        # Step 1: Run through VAD
        vad_result = self._vad.process_frame(frame)

        # Step 2: Accumulate speech frames in buffer
        if vad_result.is_speech:
            self._audio_buffer.append(frame)

        # Step 3: Check for end-of-utterance
        if self._vad.is_speech_ended() and not self._audio_buffer.is_empty():
            # End of utterance detected - process the buffered audio
            self._state.is_processing = True
            try:
                response_audio = await self._process_utterance()
                return response_audio
            finally:
                self._state.is_processing = False
                # Reset VAD and buffer for next utterance
                self._vad.reset()
                self._audio_buffer.reset()

        return None

    async def _process_utterance(self) -> bytes | None:
        """Process a complete utterance: STT → transcript → LLM → TTS → transcript.

        Flushes the audio buffer, transcribes via STT, records the agent's
        transcript entry, generates debtor response via LLM, synthesizes
        response via TTS, and records the debtor's transcript entry.

        When script_content is loaded:
        - On first utterance, delivers the opening_response from the script
        - Evaluates escalation conditions, conversation goals, trigger phrases,
          and payment conditions before LLM generation
        - Passes script_content to generate_response for prohibited response
          enforcement and emotional state transitions

        Returns:
            TTS audio bytes for the debtor response, or None on failure.
        """
        from app.services.debtor_simulator import (
            evaluate_conversation_goal_completion,
            evaluate_escalation_conditions,
            select_opening_response,
        )

        # Step 1: Flush buffer to get PCM bytes
        pcm_audio = self._audio_buffer.flush()

        if not pcm_audio:
            logger.debug("Session %s: empty audio buffer, skipping", self.session_id)
            return None

        # Step 2: Transcribe via STT
        try:
            transcription = self._stt_service.transcribe(pcm_audio)
        except Exception as e:
            logger.error(
                "Session %s: STT transcription failed: %s", self.session_id, e
            )
            return None

        if not transcription.text or not transcription.text.strip():
            logger.debug(
                "Session %s: empty transcription, skipping", self.session_id
            )
            return None

        agent_text = transcription.text.strip()
        now = datetime.now(UTC)

        # Step 3: Record agent transcript entry
        try:
            await self._transcript_manager.append_entry(
                session_id=self.session_id,
                speaker="agent",
                text=agent_text,
                timestamp=now,
            )
        except Exception as e:
            logger.error(
                "Session %s: failed to record agent transcript: %s",
                self.session_id,
                e,
            )

        # Step 3.5: First-utterance opening response from script
        if self._state.utterance_count == 0 and self._script_content is not None:
            opening = select_opening_response(self._script_content)
            if opening is not None:
                # Record opening response in transcript
                try:
                    await self._transcript_manager.append_entry(
                        session_id=self.session_id,
                        speaker="debtor",
                        text=opening,
                        timestamp=datetime.now(UTC),
                    )
                except Exception as e:
                    logger.error(
                        "Session %s: failed to record opening transcript: %s",
                        self.session_id,
                        e,
                    )

                # Synthesize opening response via TTS
                try:
                    audio_stream = await self._tts_service.synthesize(
                        opening, language="tl"
                    )
                    response_audio = audio_stream.data
                except Exception as e:
                    logger.error(
                        "Session %s: TTS synthesis for opening failed: %s",
                        self.session_id,
                        e,
                    )
                    return None

                await self._output_queue.put(response_audio)
                self._state.utterance_count += 1
                logger.info(
                    "Session %s: delivered script opening response",
                    self.session_id,
                )
                return response_audio

        # Step 3.6: Script-driven escalation and goal checks
        if self._script_content is not None:
            escalation_result = evaluate_escalation_conditions(
                agent_text, self._script_content.get("escalation_conditions")
            )
            if escalation_result is not None:
                escalation_behavior, ends_call = escalation_result
                if ends_call:
                    # Record escalation behavior in transcript
                    try:
                        await self._transcript_manager.append_entry(
                            session_id=self.session_id,
                            speaker="debtor",
                            text=escalation_behavior,
                            timestamp=datetime.now(UTC),
                        )
                    except Exception as e:
                        logger.error(
                            "Session %s: failed to record escalation transcript: %s",
                            self.session_id,
                            e,
                        )

                    # Synthesize escalation response
                    try:
                        audio_stream = await self._tts_service.synthesize(
                            escalation_behavior, language="tl"
                        )
                        response_audio = audio_stream.data
                    except Exception as e:
                        logger.error(
                            "Session %s: TTS synthesis for escalation failed: %s",
                            self.session_id,
                            e,
                        )
                        response_audio = None

                    self._state.utterance_count += 1
                    await self.teardown()
                    # Queue the sentinel after teardown; teardown clears stale
                    # output, so queuing it before teardown loses the signal.
                    await self._output_queue.put(
                        CallEndSignal(reason=escalation_behavior)
                    )
                    return response_audio

            # Check conversation goal completion
            goal_outcome = evaluate_conversation_goal_completion(
                agent_text, self._script_content.get("conversation_goal")
            )
            if goal_outcome is not None:
                completion_message = "Salamat po. Maayos na po ang usapan natin."
                try:
                    await self._transcript_manager.append_entry(
                        session_id=self.session_id,
                        speaker="debtor",
                        text=completion_message,
                        timestamp=datetime.now(UTC),
                    )
                except Exception as e:
                    logger.error(
                        "Session %s: failed to record goal completion transcript: %s",
                        self.session_id,
                        e,
                    )

                # Synthesize completion response
                try:
                    audio_stream = await self._tts_service.synthesize(
                        completion_message, language="tl"
                    )
                    response_audio = audio_stream.data
                except Exception as e:
                    logger.error(
                        "Session %s: TTS synthesis for goal completion failed: %s",
                        self.session_id,
                        e,
                    )
                    response_audio = None

                self._state.utterance_count += 1
                await self.teardown()
                # Queue after teardown so the call-ended sentinel survives
                # teardown's stale-output drain.
                await self._output_queue.put(
                    CallEndSignal(reason=goal_outcome, target_outcome=goal_outcome)
                )
                return response_audio

        # Step 4: Generate debtor response via DebtorSimulatorService
        try:
            simulator_response = await self._debtor_simulator.generate_response(
                self.persona, agent_text, script_content=self._script_content
            )
        except Exception as e:
            logger.error(
                "Session %s: debtor response generation failed: %s",
                self.session_id,
                e,
            )
            return None

        debtor_text = simulator_response.text.strip()
        response_time = datetime.now(UTC)

        # Step 5: Record debtor transcript entry
        try:
            await self._transcript_manager.append_entry(
                session_id=self.session_id,
                speaker="debtor",
                text=debtor_text,
                timestamp=response_time,
            )
        except Exception as e:
            logger.error(
                "Session %s: failed to record debtor transcript: %s",
                self.session_id,
                e,
            )

        # Step 6: Synthesize debtor response via TTS
        try:
            audio_stream = await self._tts_service.synthesize(
                debtor_text, language=simulator_response.language.lower()
            )
            response_audio = audio_stream.data
        except Exception as e:
            logger.error(
                "Session %s: TTS synthesis failed: %s", self.session_id, e
            )
            return None

        # Step 7: Queue audio for WebRTC output
        await self._output_queue.put(response_audio)
        self._state.utterance_count += 1

        logger.info(
            "Session %s: processed utterance #%d (agent: '%s' → debtor: '%s')",
            self.session_id,
            self._state.utterance_count,
            agent_text[:50],
            debtor_text[:50],
        )

        return response_audio

    async def get_next_response_audio(self) -> bytes | CallEndSignal | None:
        """Get the next queued response audio for WebRTC output.

        Returns:
            Audio bytes, a ``CallEndSignal``, or None if queue is empty.
        """
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def teardown(self) -> None:
        """Close connections, flush transcript, and clean up resources.

        Marks the pipeline as inactive, closes the WebRTC peer connection,
        and persists any buffered transcript entries.
        """
        self._state.is_active = False

        logger.info("Session %s: tearing down voice pipeline", self.session_id)

        # Close peer connection
        try:
            await self._peer_connection_manager.close_peer_connection(self.session_id)
        except Exception as e:
            logger.error(
                "Session %s: error closing peer connection: %s",
                self.session_id,
                e,
            )

        # Flush transcript to database
        try:
            await self._transcript_manager.persist(self.session_id)
        except Exception as e:
            logger.error(
                "Session %s: error persisting transcript: %s",
                self.session_id,
                e,
            )

        # Clear any remaining audio in the output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info(
            "Session %s: voice pipeline teardown complete (utterances processed: %d)",
            self.session_id,
            self._state.utterance_count,
        )
