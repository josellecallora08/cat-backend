"""Event broadcaster for dispatching real-time events to WebSocket clients."""

from datetime import UTC, datetime
from uuid import UUID

from app.schemas.event import EventData, EventMetadata, EventPayload
from app.services.event_connection_manager import EventConnectionManager
from app.services.event_store import EventStore


class EventBroadcaster:
    """Broadcasts events to connected WebSocket clients.

    Coordinates sequence number assignment, event storage, and delivery
    to all authorized connections via the ConnectionManager.
    """

    def __init__(
        self,
        connection_manager: EventConnectionManager,
        event_store: EventStore,
    ) -> None:
        """Initialize the event broadcaster.

        Args:
            connection_manager: Manages active connections and role-based delivery.
            event_store: In-memory ring buffer for gap-fill replay.
        """
        self._connection_manager = connection_manager
        self._event_store = event_store

    async def emit(
        self,
        event_type: str,
        entity_id: UUID,
        metadata: EventMetadata | None = None,
    ) -> None:
        """Broadcast an event to all authorized connections.

        Called explicitly by service methods after a successful DB commit.
        Assigns a sequence number, stores in EventStore, then dispatches
        to all connections via ConnectionManager.

        Args:
            event_type: Dot-notation event (e.g., "session.created").
            entity_id: UUID of the affected entity.
            metadata: Optional context for role-based filtering
                (campaign_id, agent_id, scenario_id).
        """
        if metadata is None:
            metadata = EventMetadata()

        seq = await self._event_store.next_seq()

        event = EventPayload(
            event=event_type,
            data=EventData(
                id=str(entity_id),
                timestamp=datetime.now(UTC).isoformat(),
            ),
            seq=seq,
        )

        self._event_store.append(event)
        await self._connection_manager.broadcast(event, metadata)
