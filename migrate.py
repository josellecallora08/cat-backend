"""Auto-migration script that runs on every deployment.

Compares the current SQLAlchemy models against the live database schema
and applies any necessary changes (new tables, new columns, type changes).

Usage:
    python migrate.py

This runs automatically before the app starts in Docker/Render deployments.
"""

import asyncio
import logging
import sys

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.database import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate")


def generate_migration_sql(conn) -> list[str]:
    """Compare models to database and generate ALTER statements."""
    # Import all models to register them with Base.metadata
    import app.models  # noqa: F401

    inspector = inspect(conn)
    statements: list[str] = []

    # 1. Create missing tables
    existing_tables = set(inspector.get_table_names())
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            # create_all will handle this, but log it
            logger.info("Table '%s' will be created", table_name)

    # 2. Add missing columns to existing tables
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

        for column in table.columns:
            if column.name in existing_columns:
                continue

            # Build column type
            col_type = column.type.compile(conn.dialect)

            # Nullable
            nullable = "NULL" if column.nullable else "NOT NULL"

            # Default value
            default = ""
            if column.server_default is not None:
                try:
                    default_text = column.server_default.arg.text
                    default = f" DEFAULT {default_text}"
                except AttributeError:
                    default_text = str(column.server_default.arg)
                    default = f" DEFAULT {default_text}"

            # If NOT NULL and no default, add a safe default to avoid errors
            if not column.nullable and not default:
                if "VARCHAR" in col_type.upper() or "TEXT" in col_type.upper():
                    default = " DEFAULT ''"
                elif "INT" in col_type.upper() or "FLOAT" in col_type.upper():
                    default = " DEFAULT 0"
                elif "BOOL" in col_type.upper():
                    default = " DEFAULT false"
                elif "TIMESTAMP" in col_type.upper() or "DATE" in col_type.upper():
                    default = " DEFAULT now()"
                elif "UUID" in col_type.upper():
                    default = " DEFAULT gen_random_uuid()"

            sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {col_type} {nullable}{default};'
            statements.append(sql)
            logger.info("+ %s.%s (%s %s%s)", table_name, column.name, col_type, nullable, default)

    # 3. Add missing indexes
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue

        existing_indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}

        for index in table.indexes:
            if index.name and index.name not in existing_indexes:
                cols = ", ".join(f'"{c.name}"' for c in index.columns)
                unique = "UNIQUE " if index.unique else ""
                sql = f'CREATE {unique}INDEX IF NOT EXISTS "{index.name}" ON "{table_name}" ({cols});'
                statements.append(sql)
                logger.info("+ index %s on %s(%s)", index.name, table_name, cols)

    return statements


def apply_migration(conn, statements: list[str]):
    """Execute migration SQL statements."""
    for sql in statements:
        logger.info("Executing: %s", sql)
        conn.execute(text(sql))


async def run_migration():
    """Run the full migration process."""
    logger.info("Starting auto-migration...")
    logger.info("Database: %s", settings.async_database_url.split("@")[-1])  # hide credentials

    engine = create_async_engine(settings.async_database_url)

    # Import models
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        # First, create any completely new tables
        await conn.run_sync(Base.metadata.create_all)

        # Then, add missing columns/indexes to existing tables
        statements = await conn.run_sync(generate_migration_sql)

        if statements:
            logger.info("Applying %d migration(s)...", len(statements))
            await conn.run_sync(apply_migration, statements)
            logger.info("Migrations applied successfully")
        else:
            logger.info("Database schema is up to date — no migrations needed")

    await engine.dispose()
    logger.info("Migration complete")


if __name__ == "__main__":
    try:
        asyncio.run(run_migration())
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        sys.exit(1)
