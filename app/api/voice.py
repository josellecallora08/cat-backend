"""WebRTC signaling WebSocket endpoint for voice sessions.

Provides WebSocket-based signaling for SDP offer/answer exchange and
ICE candidate negotiation. Uses aiortc for server-side WebRTC when available.

When a session has a pinned ScriptVersion, the voice pipeline loads the
script content and uses it for script-driven debtor behavior (opening
response, escalation, trigger phrases, etc.).

Validates: Requirements 3.4, 3.7, 9.1, 9.2, 9.3, 9.4
"""

import asyncio
import base64
import json
import logging
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from app.config import settings
from app.database import async_session_factory
from app.models import Session
from app.services.debtor_simulator import EmotionalState, PersonaContext
from app.services.llm_service import LLMService
from app.services.script_content_loader import load_script_content
from app.services.voice.peer_connection_manager import (
    AIORTC_AVAILABLE,
    PeerConnectionManager,
)
from app.services.voice.pipeline_factory import create_voice_pipeline
from app.services.voice.voice_pipeline import CallEndSignal


logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton peer connection manager shared across WebSocket connections
_peer_connection_manager = PeerConnectionManager()


def get_peer_connection_manager() -> PeerConnectionManager:
    """Get the shared PeerConnectionManager instance."""
    return _peer_connection_manager


@router.websocket("/ws/voice/{session_id}")
async def voice_signaling_websocket(websocket: WebSocket, session_id: UUID) -> None:
    """WebSocket endpoint for WebRTC signaling.

    Handles JSON messages with the following types:
    - "offer": Contains SDP offer for WebRTC negotiation
      Payload: {"type": "offer", "sdp": "<SDP string>"}
    - "ice_candidate": Contains an ICE candidate
      Payload: {"type": "ice_candidate", "candidate": "<candidate string>",
                "sdpMid": "<media id>", "sdpMLineIndex": <index>}

    Responses:
    - "answer": SDP answer after processing an offer
      Payload: {"type": "answer", "sdp": "<SDP string>"}
    - "error": Error message
      Payload: {"type": "error", "message": "<error description>"}
    """
    await websocket.accept()

    # --- JWT Authentication ---
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        return

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id_str = payload.get("sub")
        if not user_id_str:
            await websocket.close(code=4401)
            return
    except JWTError:
        await websocket.close(code=4401)
        return

    manager = get_peer_connection_manager()
    db_context = async_session_factory()
    db = await db_context.__aenter__()

    # Resolve the immutable script snapshot once per connection.  The
    # signaling route is intentionally transport-only today, but keeping the
    # pinned content on the connection makes it available to the pipeline
    # attachment when media handling is installed and prevents looking up a
    # mutable Script.current_version_id mid-call.
    script_content = None
    session = None
    try:
        session = await db.get(Session, session_id)
        if session is not None:
            script_content = await load_script_content(db, session.script_version_id)
    except Exception as exc:
        # Signaling must remain available when the optional database is
        # unavailable; media still works with the legacy unscripted path.
        logger.warning("Voice session lookup unavailable: %s", exc)
    websocket.state.script_content = script_content

    if not AIORTC_AVAILABLE:
        await websocket.send_json(
            {
                "type": "error",
                "message": "WebRTC is not available: aiortc is not installed on the server.",
            }
        )
        await websocket.close(code=1011, reason="aiortc not available")
        await db_context.__aexit__(None, None, None)
        return

    pipeline = None
    output_task = None
    if session is not None:
        persona_data = session.persona_context or {}
        pipeline = create_voice_pipeline(
            session_id=session_id,
            persona=PersonaContext(
                persona_id=session_id,
                name=persona_data.get("name", "Debtor"),
                communication_style=persona_data.get("communication_style", "cooperative"),
                financial_circumstances=persona_data.get("financial_circumstances", {}),
                emotional_state=EmotionalState(persona_data.get("emotional_state", 3)),
                language=persona_data.get("language", "TAGLISH"),
            ),
            db=db,
            llm_service=LLMService(),
            peer_connection_manager=manager,
            script_content=script_content,
        )

        async def forward_output():
            while True:
                item = await pipeline.get_next_response_audio()
                if item is None:
                    await asyncio.sleep(0.01)
                    continue
                if isinstance(item, CallEndSignal):
                    await websocket.send_json(
                        {
                            "type": "call_ended",
                            "reason": item.reason,
                            "target_outcome": item.target_outcome,
                        }
                    )
                    await websocket.close()
                    return
                await websocket.send_json(
                    {
                        "type": "audio",
                        "audio": base64.b64encode(item).decode("ascii"),
                    }
                )

        output_task = asyncio.create_task(forward_output())

    logger.info(f"Voice WebSocket connected for session {session_id}")

    try:
        while True:
            # Receive JSON message from client
            raw_data = await websocket.receive_text()

            try:
                message = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON message"})
                continue

            msg_type = message.get("type")

            if msg_type == "offer":
                # Handle SDP offer
                sdp = message.get("sdp")
                if not sdp:
                    await websocket.send_json(
                        {"type": "error", "message": "Missing 'sdp' in offer message"}
                    )
                    continue

                try:
                    answer = await manager.handle_offer(
                        session_id=session_id,
                        sdp=sdp,
                        sdp_type="offer",
                        on_track=pipeline.handle_audio_track if pipeline else None,
                    )
                    await websocket.send_json({"type": "answer", "sdp": answer["sdp"]})
                    logger.info(f"Session {session_id}: SDP offer processed, answer sent")
                except Exception as e:
                    logger.error(f"Session {session_id}: error handling offer: {e}")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Failed to process offer: {e!s}",
                        }
                    )

            elif msg_type == "ice_candidate":
                raw_candidate = message.get("candidate")

                if isinstance(raw_candidate, dict):
                    # Nested format: RTCIceCandidate.toJSON() output
                    candidate_str = raw_candidate.get("candidate")
                    sdp_mid = raw_candidate.get("sdpMid")
                    sdp_mline_index = raw_candidate.get("sdpMLineIndex")
                elif isinstance(raw_candidate, str):
                    # Flat format: backward-compatible plain string
                    candidate_str = raw_candidate
                    sdp_mid = message.get("sdpMid")
                    sdp_mline_index = message.get("sdpMLineIndex")
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid ICE candidate format: "
                            "'candidate' must be a string or object",
                        }
                    )
                    continue

                if not candidate_str:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Missing 'candidate' value in ice_candidate message",
                        }
                    )
                    continue

                try:
                    await manager.add_ice_candidate(
                        session_id=session_id,
                        candidate=candidate_str,
                        sdp_mid=sdp_mid,
                        sdp_mline_index=sdp_mline_index,
                    )
                    await websocket.send_json({"type": "ice_candidate_ack", "status": "added"})
                except Exception as e:
                    logger.error(f"Session {session_id}: error adding ICE candidate: {e}")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Failed to add ICE candidate: {e!s}",
                        }
                    )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error for session {session_id}: {e}")
    finally:
        if output_task is not None:
            output_task.cancel()
        if pipeline is not None:
            await pipeline.teardown()
        await db_context.__aexit__(None, None, None)
        # Clean up the peer connection when WebSocket closes
        await manager.close_peer_connection(session_id)
