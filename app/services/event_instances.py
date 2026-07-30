"""Singleton instances for the real-time event system."""

from app.services.event_broadcaster import EventBroadcaster
from app.services.event_connection_manager import EventConnectionManager
from app.services.event_store import EventStore


event_store = EventStore(ttl_seconds=300)
event_connection_manager = EventConnectionManager()
event_broadcaster = EventBroadcaster(
    connection_manager=event_connection_manager,
    event_store=event_store,
)
