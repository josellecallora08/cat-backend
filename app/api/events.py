"""WebSocket endpoint for real-time event streaming."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.models.campaign import CampaignAgent
from app.models.user import User
from app.schemas.event import EventData, EventPayload
from app.services.event_instances import event_connection_manager, event_store


logger = logging.getLogger(__name__)

router = APIRouter()

HEARTBEAT_INTERVAL = 20
HEARTBEAT_TIMEOUT = 30


async def _heartbeat_loop(websocket: WebSocket) -> None:
    """Send periodic pings and detect unresponsive clients.

    Args:
        websocket: The active WebSocket connection to monitor.
    """
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await asyncio.wait_for(
                websocket.send_json({"ping": True}),
                timeout=HEARTBEAT_TIMEOUT,
            )
    except (asyncio.TimeoutError, WebSocketDisconnect, Exception):  # noqa: BLE001
        pass


@router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket) -> None:
    """Real-time event streaming endpoint.

    Authenticates the connection via JWT token query parameter, registers
    the client with the connection manager, handles gap-fill replay, and
    maintains the connection with periodic heartbeat pings.

    Query params:
        token: JWT access token (required).
        last_seq: Last received sequence number for gap-fill (optional).
    """
    await websocket.accept()

    # Extract token from query params
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        return

    # Validate JWT
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id_str = payload.get("sub")
        if not user_id_str:
            await websocket.close(code=4401)
            return
    except JWTError:
        await websocket.close(code=4401)
        return

    # Look up user in DB
    async with async_session_factory() as db:
        stmt = select(User).where(User.id == user_id_str)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            await websocket.close(code=4403)
            return

        # Get user's campaign_ids
        campaign_stmt = select(CampaignAgent.campaign_id).where(
            CampaignAgent.agent_id == user.id
        )
        campaign_result = await db.execute(campaign_stmt)
        campaign_ids = {row[0] for row in campaign_result.all()}

    # Register connection with ConnectionManager
    await event_connection_manager.connect(
        websocket=websocket,
        user_id=user.id,
        role=user.role,
        user_type=user.user_type,
        campaign_ids=campaign_ids,
    )

    # Handle gap-fill replay
    last_seq_param = websocket.query_params.get("last_seq")
    if last_seq_param is not None:
        try:
            last_seq = int(last_seq_param)
            missed = event_store.get_after(last_seq)
            if missed is None:
                # Gap too large — send resync signal
                resync_payload = EventPayload(
                    event="system.resync",
                    data=EventData(
                        id="0",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    seq=await event_store.next_seq(),
                )
                await websocket.send_json(resync_payload.model_dump())
            else:
                for event in missed:
                    await websocket.send_json(event.model_dump())
        except (ValueError, TypeError):
            pass

    # Start heartbeat background task
    heartbeat_task = asyncio.create_task(_heartbeat_loop(websocket))

    try:
        # Receive loop — keep connection alive and listen for client messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
        await event_connection_manager.disconnect(websocket)
