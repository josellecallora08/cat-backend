"""WebSocket connection manager with role-based event filtering."""

import logging
from dataclasses import dataclass, field
from uuid import UUID

from fastapi import WebSocket

from app.schemas.event import EventMetadata, EventPayload


logger = logging.getLogger(__name__)


@dataclass
class ConnectedClient:
    """Metadata for a connected WebSocket client.

    Attributes:
        websocket: The active WebSocket connection.
        user_id: The authenticated user's unique identifier.
        role: The user's role (e.g., "admin", "user").
        user_type: The user's type within their role (e.g., "trainer", "agent").
        campaign_ids: Set of campaign UUIDs the user is assigned to.
    """

    websocket: WebSocket
    user_id: UUID
    role: str
    user_type: str | None
    campaign_ids: set[UUID] = field(default_factory=set)


class EventConnectionManager:
    """Manages active WebSocket connections and role-based delivery.

    Tracks all connected clients, applies role-based filtering when
    broadcasting events, and handles connection lifecycle operations.
    """

    def __init__(self) -> None:
        """Initialize the connection manager with an empty client registry."""
        self._clients: dict[WebSocket, ConnectedClient] = {}

    async def connect(
        self,
        websocket: WebSocket,
        user_id: UUID,
        role: str,
        user_type: str | None,
        campaign_ids: set[UUID],
    ) -> None:
        """Register a new client connection.

        Args:
            websocket: The WebSocket connection to register.
            user_id: The authenticated user's ID.
            role: The user's role.
            user_type: The user's type (trainer, agent, or None).
            campaign_ids: Campaign UUIDs assigned to this user.
        """
        client = ConnectedClient(
            websocket=websocket,
            user_id=user_id,
            role=role,
            user_type=user_type,
            campaign_ids=campaign_ids,
        )
        self._clients[websocket] = client
        logger.info(
            "Client connected: user_id=%s, role=%s, user_type=%s",
            user_id,
            role,
            user_type,
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a client connection and release resources.

        Args:
            websocket: The WebSocket connection to remove.
        """
        client = self._clients.pop(websocket, None)
        if client:
            logger.info("Client disconnected: user_id=%s", client.user_id)

    async def broadcast(self, event: EventPayload, metadata: EventMetadata) -> None:
        """Send event to all authorized clients based on role filtering.

        Iterates over connected clients, checks delivery authorization via
        _should_deliver, and sends the JSON payload. If sending fails, the
        client is disconnected.

        Args:
            event: The event payload to broadcast.
            metadata: Routing metadata for filtering decisions.
        """
        disconnected: list[WebSocket] = []

        for websocket, client in self._clients.items():
            if not self._should_deliver(client, event, metadata):
                continue

            try:
                await websocket.send_json(event.model_dump())
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to send event to user_id=%s, disconnecting",
                    client.user_id,
                )
                disconnected.append(websocket)

        for websocket in disconnected:
            await self.disconnect(websocket)

    async def update_client_scope(
        self,
        user_id: UUID,
        campaign_ids: set[UUID],
    ) -> None:
        """Update campaign assignments for a connected client.

        Args:
            user_id: The user whose scope should be updated.
            campaign_ids: The new set of campaign UUIDs.
        """
        for client in self._clients.values():
            if client.user_id == user_id:
                client.campaign_ids = campaign_ids

    async def disconnect_user(self, user_id: UUID, code: int) -> None:
        """Force-disconnect all connections for a specific user.

        Args:
            user_id: The user to disconnect.
            code: The WebSocket close code to send.
        """
        to_remove: list[WebSocket] = []

        for websocket, client in self._clients.items():
            if client.user_id == user_id:
                to_remove.append(websocket)

        for websocket in to_remove:
            try:
                await websocket.close(code=code)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Error closing connection for user_id=%s",
                    user_id,
                )
            await self.disconnect(websocket)

    async def close_all(self, code: int = 1001) -> None:
        """Gracefully close all active connections.

        Used during application shutdown to notify clients.

        Args:
            code: The WebSocket close code to send. Defaults to 1001 (Going Away).
        """
        for websocket, client in list(self._clients.items()):
            try:
                await websocket.close(code=code)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Error closing connection for user_id=%s during shutdown",
                    client.user_id,
                )

        self._clients.clear()
        logger.info("All connections closed")

    def _should_deliver(
        self,
        client: ConnectedClient,
        event: EventPayload,
        meta: EventMetadata,
    ) -> bool:
        """Determine if a client should receive this event.

        Applies role-based filtering rules:
        - Admins receive all events.
        - Non-admins never receive user-category or dashboard-category events.
        - Trainers receive events scoped to their assigned campaigns.
        - Agents receive events for their own sessions or campaign-scoped
          scenario/campaign events.

        Args:
            client: The connected client's metadata.
            event: The event payload.
            meta: Routing metadata (campaign_id, agent_id, scenario_id).

        Returns:
            True if the client is authorized to receive this event.
        """
        # Admins receive everything
        if client.role == "admin":
            return True

        # Extract event category from dot-notation type
        category = event.event.split(".")[0]

        # Non-admins never receive user-category events
        if category == "user":
            return False

        # Dashboard events go to admins only (handled above)
        if category == "dashboard":
            return False

        # Trainer: receives events for their assigned campaigns
        if client.user_type == "trainer":
            if meta.campaign_id and meta.campaign_id in client.campaign_ids:
                return True
            return False

        # Agent: receives events for own sessions or own campaign scenarios
        if client.user_type == "agent":
            if meta.agent_id and meta.agent_id == client.user_id:
                return True
            if meta.campaign_id and meta.campaign_id in client.campaign_ids:
                if category in ("scenario", "campaign"):
                    return True
            return False

        return False
