"""Utility to ensure the PostgreSQL database exists before starting the app."""

import logging
from urllib.parse import urlparse

import asyncpg


logger = logging.getLogger(__name__)


async def ensure_database_exists(database_url: str) -> None:
    """Create the PostgreSQL database if it does not already exist.

    Connects to the default 'postgres' maintenance database to check whether
    the target database exists, and issues CREATE DATABASE if it's missing.

    For non-PostgreSQL URLs (e.g. SQLite), this function is a no-op.

    Args:
        database_url: The full async database URL (e.g. postgresql+asyncpg://...).
    """
    if "postgresql" not in database_url and "postgres" not in database_url:
        logger.debug("Non-PostgreSQL database URL detected, skipping auto-creation")
        return

    parsed = urlparse(database_url)
    db_name = parsed.path.lstrip("/")

    if not db_name:
        logger.warning(
            "Could not extract database name from URL, skipping auto-creation"
        )
        return

    # Build a connection string pointing to the default 'postgres' database
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = parsed.username or "postgres"
    password = parsed.password

    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",
        )
    except Exception:
        logger.warning(
            "Could not connect to 'postgres' maintenance database. "
            "Skipping auto-creation — the target database may already exist.",
            exc_info=True,
        )
        return

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            db_name,
        )

        if not exists:
            # CREATE DATABASE cannot run inside a transaction
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info("Created database '%s'", db_name)
        else:
            logger.debug("Database '%s' already exists", db_name)
    except asyncpg.exceptions.DuplicateDatabaseError:
        # Race condition: another process created it between our check and create
        logger.debug("Database '%s' was created by another process", db_name)
    finally:
        await conn.close()
