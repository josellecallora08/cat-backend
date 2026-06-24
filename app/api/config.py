"""System configuration API endpoints (admin only).

Provides get/update endpoints for global system settings like the
debtor simulator prompt configuration.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import SystemConfig
from app.models.user import User
from app.services.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Schemas ---


class ConfigValue(BaseModel):
    key: str
    value: str


class UpdateConfigRequest(BaseModel):
    value: str


class GlobalPromptResponse(BaseModel):
    global_prompt: str


# --- Well-known config keys ---

GLOBAL_DEBTOR_PROMPT_KEY = "global_debtor_prompt"

DEFAULT_GLOBAL_PROMPT = """ADDITIONAL ADMIN INSTRUCTIONS:
- Follow all company policies during the call.
- Be respectful but stay in character as the debtor persona.
- Do not provide any real personal information."""


# --- Endpoints ---


@router.get("/config/{key}", response_model=ConfigValue)
async def get_config(
    key: str,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
):
    """Get a system configuration value by key (admin only)."""
    stmt = select(SystemConfig).where(SystemConfig.key == key)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()

    if config is None:
        # Return default for known keys
        if key == GLOBAL_DEBTOR_PROMPT_KEY:
            return ConfigValue(key=key, value=DEFAULT_GLOBAL_PROMPT)
        raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")

    return ConfigValue(key=config.key, value=config.value)


@router.put("/config/{key}", response_model=ConfigValue)
async def update_config(
    key: str,
    body: UpdateConfigRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
):
    """Update a system configuration value (admin only). Creates if not exists."""
    stmt = select(SystemConfig).where(SystemConfig.key == key)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()

    if config is None:
        config = SystemConfig(key=key, value=body.value)
        db.add(config)
    else:
        config.value = body.value

    await db.commit()
    await db.refresh(config)

    logger.info("Config updated: %s (by %s)", key, admin.email)

    return ConfigValue(key=config.key, value=config.value)


@router.get("/config/prompt/global", response_model=GlobalPromptResponse)
async def get_global_prompt(
    db: AsyncSession = Depends(get_session),
):
    """Get the global debtor prompt (no auth required — used by simulator)."""
    stmt = select(SystemConfig).where(SystemConfig.key == GLOBAL_DEBTOR_PROMPT_KEY)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()

    return GlobalPromptResponse(
        global_prompt=config.value if config else DEFAULT_GLOBAL_PROMPT
    )
