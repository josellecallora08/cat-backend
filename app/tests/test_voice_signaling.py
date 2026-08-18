"""Tests for WebRTC signaling WebSocket endpoint.

Tests that the WebSocket accepts connections and handles message formats
correctly. Since aiortc may not be installed in the test environment,
these tests verify signaling protocol behavior with appropriate mocking.

Validates: Requirements 3.4, 3.7, 5.1, 5.2, 5.3, 5.4, 5.5
"""

# Test-client context nesting and broad connection exceptions are intentional.

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from jose import jwt
from starlette.testclient import TestClient

from app.config import settings
from app.main import app


def _make_token(sub: str | None = None, secret: str | None = None) -> str:
    """Create a JWT token for testing voice WebSocket auth."""
    payload = {}
    if sub is not None:
        payload["sub"] = sub
    return jwt.encode(payload, secret or settings.jwt_secret, algorithm="HS256")


@pytest.fixture
def valid_token():
    """A valid JWT token with a sub claim."""
    return _make_token(sub=str(uuid4()))


@pytest.fixture
def test_client():
    """Synchronous test client for WebSocket testing."""
    return TestClient(app)


@pytest.fixture
def session_id():
    """Generate a test session ID."""
    return str(uuid4())


class TestVoiceWebSocketConnection:
    """Test WebSocket connection handling."""

    def test_websocket_accepts_connection(self, test_client, session_id, valid_token):
        """WebSocket endpoint should accept connections with a valid token."""
        with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
            # Connection accepted - if aiortc is not available, we get an error message
            data = ws.receive_json()
            # Either we get an error about aiortc not being available (expected in test env)
            # or the connection stays open for signaling
            if data.get("type") == "error":
                assert "aiortc" in data["message"].lower() or "webrtc" in data["message"].lower()

    def test_websocket_rejects_invalid_session_id(self, test_client):
        """WebSocket should reject invalid session IDs (non-UUID)."""
        with pytest.raises(Exception):
            with test_client.websocket_connect("/ws/voice/not-a-uuid") as _ws:
                pass


class TestVoiceWebSocketSignaling:
    """Test WebSocket signaling protocol with mocked aiortc."""

    @patch("app.api.voice.AIORTC_AVAILABLE", True)
    @patch("app.api.voice.get_peer_connection_manager")
    def test_offer_message_returns_answer(
        self, mock_get_manager, test_client, session_id, valid_token
    ):
        """Sending an SDP offer should return an SDP answer."""
        mock_manager = MagicMock()
        mock_manager.is_available = True
        mock_manager.handle_offer = AsyncMock(return_value={"sdp": "v=0\r\n...", "type": "answer"})
        mock_manager.close_peer_connection = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Also patch the module-level AIORTC_AVAILABLE used in the websocket handler
        with patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", True):
            with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
                # Send SDP offer
                ws.send_json({"type": "offer", "sdp": "v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\n..."})
                response = ws.receive_json()
                assert response["type"] == "answer"
                assert "sdp" in response

    @patch("app.api.voice.AIORTC_AVAILABLE", True)
    @patch("app.api.voice.get_peer_connection_manager")
    def test_ice_candidate_message_acknowledged(
        self, mock_get_manager, test_client, session_id, valid_token
    ):
        """Sending an ICE candidate should return an acknowledgment."""
        mock_manager = MagicMock()
        mock_manager.is_available = True
        mock_manager.add_ice_candidate = AsyncMock()
        mock_manager.close_peer_connection = AsyncMock()
        mock_get_manager.return_value = mock_manager

        with patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", True):
            with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
                # Send ICE candidate
                ws.send_json(
                    {
                        "type": "ice_candidate",
                        "candidate": "candidate:1 1 UDP 2130706431 192.168.1.1 12345 typ host",
                        "sdpMid": "audio",
                        "sdpMLineIndex": 0,
                    }
                )
                response = ws.receive_json()
                assert response["type"] == "ice_candidate_ack"
                assert response["status"] == "added"

    @patch("app.api.voice.AIORTC_AVAILABLE", True)
    @patch("app.api.voice.get_peer_connection_manager")
    def test_invalid_json_returns_error(
        self, mock_get_manager, test_client, session_id, valid_token
    ):
        """Sending invalid JSON should return an error message."""
        mock_manager = MagicMock()
        mock_manager.is_available = True
        mock_manager.close_peer_connection = AsyncMock()
        mock_get_manager.return_value = mock_manager

        with patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", True):
            with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
                ws.send_text("not valid json {{{")
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "invalid json" in response["message"].lower()

    @patch("app.api.voice.AIORTC_AVAILABLE", True)
    @patch("app.api.voice.get_peer_connection_manager")
    def test_unknown_message_type_returns_error(
        self, mock_get_manager, test_client, session_id, valid_token
    ):
        """Sending an unknown message type should return an error."""
        mock_manager = MagicMock()
        mock_manager.is_available = True
        mock_manager.close_peer_connection = AsyncMock()
        mock_get_manager.return_value = mock_manager

        with patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", True):
            with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
                ws.send_json({"type": "unknown_type", "data": "something"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "unknown message type" in response["message"].lower()

    @patch("app.api.voice.AIORTC_AVAILABLE", True)
    @patch("app.api.voice.get_peer_connection_manager")
    def test_offer_without_sdp_returns_error(
        self, mock_get_manager, test_client, session_id, valid_token
    ):
        """Sending an offer without SDP should return an error."""
        mock_manager = MagicMock()
        mock_manager.is_available = True
        mock_manager.close_peer_connection = AsyncMock()
        mock_get_manager.return_value = mock_manager

        with patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", True):
            with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
                ws.send_json({"type": "offer"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "sdp" in response["message"].lower()

    @patch("app.api.voice.AIORTC_AVAILABLE", True)
    @patch("app.api.voice.get_peer_connection_manager")
    def test_ice_candidate_without_candidate_returns_error(
        self, mock_get_manager, test_client, session_id, valid_token
    ):
        """Sending an ICE candidate without the candidate string should return an error."""
        mock_manager = MagicMock()
        mock_manager.is_available = True
        mock_manager.close_peer_connection = AsyncMock()
        mock_get_manager.return_value = mock_manager

        with patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", True):
            with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
                ws.send_json({"type": "ice_candidate"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "candidate" in response["message"].lower()


class TestVoiceWebSocketWithoutAiortc:
    """Test behavior when aiortc is not installed."""

    @patch("app.api.voice.AIORTC_AVAILABLE", False)
    def test_returns_error_when_aiortc_unavailable(self, test_client, session_id, valid_token):
        """When aiortc is not installed, WebSocket should report the error and close."""
        with test_client.websocket_connect(f"/ws/voice/{session_id}?token={valid_token}") as ws:
            response = ws.receive_json()
            assert response["type"] == "error"
            assert (
                "not available" in response["message"].lower()
                or "not installed" in response["message"].lower()
            )


class TestPeerConnectionManager:
    """Test PeerConnectionManager unit behavior."""

    def test_is_available_reflects_aiortc_import(self):
        """is_available should reflect whether aiortc was imported."""
        from app.services.voice.peer_connection_manager import (
            AIORTC_AVAILABLE,
            PeerConnectionManager,
        )

        manager = PeerConnectionManager()
        assert manager.is_available == AIORTC_AVAILABLE

    @pytest.mark.asyncio
    async def test_create_peer_connection_raises_without_aiortc(self):
        """create_peer_connection should raise RuntimeError if aiortc is not available."""
        from app.services.voice.peer_connection_manager import PeerConnectionManager

        manager = PeerConnectionManager()
        with (
            patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", False),
            pytest.raises(RuntimeError, match="aiortc is not installed"),
        ):
            await manager.create_peer_connection(uuid4())

    @pytest.mark.asyncio
    async def test_handle_offer_raises_without_aiortc(self):
        """handle_offer should raise RuntimeError if aiortc is not available."""
        from app.services.voice.peer_connection_manager import PeerConnectionManager

        manager = PeerConnectionManager()
        with (
            patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", False),
            pytest.raises(RuntimeError, match="aiortc is not installed"),
        ):
            await manager.handle_offer(uuid4(), "v=0\r\n...")

    @pytest.mark.asyncio
    async def test_add_ice_candidate_raises_without_aiortc(self):
        """add_ice_candidate should raise RuntimeError if aiortc is not available."""
        from app.services.voice.peer_connection_manager import PeerConnectionManager

        manager = PeerConnectionManager()
        with (
            patch("app.services.voice.peer_connection_manager.AIORTC_AVAILABLE", False),
            pytest.raises(RuntimeError, match="aiortc is not installed"),
        ):
            await manager.add_ice_candidate(uuid4(), "candidate:1...")
