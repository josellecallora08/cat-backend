"""Authorization policy for viewing session details and generated artifacts."""

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Session
from app.models.user import User, UserRole, UserType
from app.services.session_service import get_session
from app.services.trainer_service import (
    get_trainer_campaign,
    get_trainer_campaign_agent_ids,
)


async def get_authorized_session(
    db: AsyncSession,
    session_id: UUID,
    current_user: User,
) -> Session:
    """Return a session when the authenticated user may view it.

    Access is granted to administrators, the owning agent, or a trainer whose
    active campaign includes the session owner. The session is loaded before
    authorization so missing sessions consistently return 404.
    """
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if current_user.role == UserRole.ADMIN.value:
        return session

    if (
        current_user.role == UserRole.USER.value
        and current_user.user_type == UserType.TRAINER.value
    ):
        campaign = await get_trainer_campaign(db, current_user.id)
        if campaign is not None:
            campaign_agent_ids = await get_trainer_campaign_agent_ids(db, campaign.id)
            if session.agent_id in campaign_agent_ids:
                return session
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session access denied",
        )

    if session.agent_id == current_user.id:
        return session

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Session access denied",
    )
