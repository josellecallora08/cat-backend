"""Repository layer for Scenario CRUD operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Scenario


async def list_active_scenarios(db: AsyncSession) -> List[Scenario]:
    """List all active scenarios ordered by name.

    Returns only scenarios where is_active is True, sorted alphabetically by name.
    """
    stmt = (
        select(Scenario)
        .where(Scenario.is_active == True)  # noqa: E712
        .order_by(Scenario.name)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_scenario_by_id(db: AsyncSession, scenario_id: UUID) -> Optional[Scenario]:
    """Get a scenario by ID.

    Returns None if the scenario does not exist or is inactive.
    """
    stmt = select(Scenario).where(
        Scenario.id == scenario_id,
        Scenario.is_active == True,  # noqa: E712
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
