"""Pydantic schemas for real-time WebSocket event payloads and metadata."""

from uuid import UUID

from pydantic import BaseModel


class EventData(BaseModel):
    """Payload data section of an event.

    Contains the entity identifier and the timestamp of when the event occurred.
    """

    id: str
    timestamp: str


class EventPayload(BaseModel):
    """Complete event payload sent over WebSocket.

    Represents the JSON structure delivered to connected clients, including
    the event type in dot-notation, the data payload, and a monotonically
    increasing sequence number for gap-fill replay.
    """

    event: str
    data: EventData
    seq: int


class EventMetadata(BaseModel):
    """Routing metadata for role-based filtering (not sent to client).

    Used by the ConnectionManager to determine which connected clients
    should receive a given event based on their role and campaign assignments.
    """

    campaign_id: UUID | None = None
    agent_id: UUID | None = None
    scenario_id: UUID | None = None
