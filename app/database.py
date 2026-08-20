from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


engine = create_async_engine(
    settings.async_database_url,
    # SQL statements are extremely verbose and can contain sensitive values.
    # Keep them independent from general application debug mode and opt-in only.
    echo=settings.database_echo,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncSession:
    """Dependency that provides an async database session."""
    async with async_session_factory() as session:
        yield session
