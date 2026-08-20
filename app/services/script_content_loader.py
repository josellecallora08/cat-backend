"""Script content loader for call-script integration.

Provides a helper to load the pinned ScriptVersion content (ScriptContract)
from the database, used by both the text call endpoint and voice pipeline.

Validates: Requirements 1.1, 1.3
"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.script import ScriptVersion


async def load_script_content(
    db: AsyncSession, script_version_id: UUID | None
) -> dict[str, Any] | None:
    """Load the ScriptContract content dict for a pinned ScriptVersion.

    Args:
        db: Async database session.
        script_version_id: The session's pinned script version ID, or None.

    Returns:
        The ScriptVersion.content dict (ScriptContract), or None if
        script_version_id is None or the version doesn't exist.
    """
    if script_version_id is None:
        return None

    stmt = select(ScriptVersion.content).where(ScriptVersion.id == script_version_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    return row  # Already a dict from JSONB column
