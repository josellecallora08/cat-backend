"""WebRTC signaling WebSocket endpoint for voice sessions.

Provides WebSocket-based signaling for SDP offer/answer exchange and
ICE candidate negotiation. Uses aiortc for server-side WebRTC when available.

Validates: Requirements 3.4, 3.7
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.voice.peer_connection_manager import (
    PeerConnectionManager,
    AIORTC_AVAILABLE,
)

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

    # Check if aiortc is available
    if not AIORTC_AVAILABLE:
        await websocket.send_json(
            {
                "type": "error",
                "message": "WebRTC is not available: aiortc is not installed on the server.",
            }
        )
        await websocket.close(code=1011, reason="aiortc not available")
        return

    manager = get_peer_connection_manager()

    logger.info(f"Voice WebSocket connected for session {session_id}")

    try:
        while True:
            # Receive JSON message from client
            raw_data = await websocket.receive_text()

            try:
                message = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON message"}
                )
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
                    )
                    await websocket.send_json(
                        {"type": "answer", "sdp": answer["sdp"]}
                    )
                    logger.info(
                        f"Session {session_id}: SDP offer processed, answer sent"
                    )
                except Exception as e:
                    logger.error(
                        f"Session {session_id}: error handling offer: {e}"
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Failed to process offer: {str(e)}",
                        }
                    )

            elif msg_type == "ice_candidate":
                # Handle ICE candidate
                candidate = message.get("candidate")
                if not candidate:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Missing 'candidate' in ice_candidate message",
                        }
                    )
                    continue

                try:
                    await manager.add_ice_candidate(
                        session_id=session_id,
                        candidate=candidate,
                        sdp_mid=message.get("sdpMid"),
                        sdp_mline_index=message.get("sdpMLineIndex"),
                    )
                    await websocket.send_json(
                        {"type": "ice_candidate_ack", "status": "added"}
                    )
                except Exception as e:
                    logger.error(
                        f"Session {session_id}: error adding ICE candidate: {e}"
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Failed to add ICE candidate: {str(e)}",
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
        # Clean up the peer connection when WebSocket closes
        await manager.close_peer_connection(session_id)
