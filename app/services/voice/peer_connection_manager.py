"""Manages WebRTC peer connections for voice sessions.

Uses aiortc for server-side WebRTC when available. Falls back gracefully
if aiortc is not installed.

Validates: Requirements 3.4, 3.7
"""

import logging
from uuid import UUID
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Conditional aiortc import
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.contrib.media import MediaRelay

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCIceCandidate = None
    MediaRelay = None


class PeerConnectionManager:
    """Manages WebRTC peer connections for active voice sessions.

    Each session gets its own RTCPeerConnection instance. The manager
    handles creation, ICE candidate exchange, and teardown.
    """

    def __init__(self) -> None:
        self._connections: dict[UUID, Any] = {}
        self._relay = MediaRelay() if AIORTC_AVAILABLE and MediaRelay else None

    @property
    def is_available(self) -> bool:
        """Check if aiortc is available for WebRTC connections."""
        return AIORTC_AVAILABLE

    async def create_peer_connection(self, session_id: UUID) -> Any:
        """Create a new RTCPeerConnection for the given session.

        Returns the peer connection instance.
        Raises RuntimeError if aiortc is not available.
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc is not installed. WebRTC functionality is unavailable."
            )

        pc = RTCPeerConnection()
        self._connections[session_id] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info(
                f"Session {session_id}: connection state is {pc.connectionState}"
            )
            if pc.connectionState == "failed":
                await self.close_peer_connection(session_id)

        @pc.on("track")
        async def on_track(track: Any) -> None:
            logger.info(
                f"Session {session_id}: received {track.kind} track"
            )
            # Audio track handling will be wired in VoicePipelineOrchestrator (task 13.2)

        logger.info(f"Session {session_id}: peer connection created")
        return pc

    async def handle_offer(
        self, session_id: UUID, sdp: str, sdp_type: str = "offer"
    ) -> dict:
        """Process an SDP offer and return an SDP answer.

        Args:
            session_id: The session UUID.
            sdp: The SDP offer string from the client.
            sdp_type: The type of SDP message (usually "offer").

        Returns:
            Dict with 'sdp' and 'type' keys for the answer.
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc is not installed. WebRTC functionality is unavailable."
            )

        pc = self._connections.get(session_id)
        if pc is None:
            pc = await self.create_peer_connection(session_id)

        # Set the remote description from the client's offer
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await pc.setRemoteDescription(offer)

        # Create and set the local answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def add_ice_candidate(
        self, session_id: UUID, candidate: str, sdp_mid: Optional[str] = None,
        sdp_mline_index: Optional[int] = None
    ) -> None:
        """Add an ICE candidate received from the client.

        Args:
            session_id: The session UUID.
            candidate: The ICE candidate string.
            sdp_mid: The media stream identification tag.
            sdp_mline_index: The index of the media description.
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc is not installed. WebRTC functionality is unavailable."
            )

        pc = self._connections.get(session_id)
        if pc is None:
            logger.warning(
                f"Session {session_id}: received ICE candidate but no peer connection exists"
            )
            return

        # aiortc handles ICE candidates via addIceCandidate
        ice_candidate = RTCIceCandidate(
            sdpMid=sdp_mid or "",
            sdpMLineIndex=sdp_mline_index or 0,
            candidate=candidate,
        )
        await pc.addIceCandidate(ice_candidate)
        logger.info(f"Session {session_id}: added ICE candidate")

    async def close_peer_connection(self, session_id: UUID) -> None:
        """Close and clean up the peer connection for a session."""
        pc = self._connections.pop(session_id, None)
        if pc is not None:
            await pc.close()
            logger.info(f"Session {session_id}: peer connection closed")

    def get_connection(self, session_id: UUID) -> Optional[Any]:
        """Get the active peer connection for a session, if any."""
        return self._connections.get(session_id)

    async def close_all(self) -> None:
        """Close all active peer connections."""
        session_ids = list(self._connections.keys())
        for session_id in session_ids:
            await self.close_peer_connection(session_id)
