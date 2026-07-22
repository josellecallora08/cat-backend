import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    campaigns,
    scenarios,
    sessions,
    voice,
    tts,
    dashboard,
    auth,
    config,
    campaign_dashboard,
)
from app.config import settings
from app.database import async_session_factory, get_session

logger = logging.getLogger(__name__)


def _run_migrations() -> None:
    """Run Alembic migrations to ensure the database schema is up to date."""
    import shutil
    import subprocess
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    # Find the alembic executable (installed via pip)
    alembic_bin = shutil.which("alembic")
    if not alembic_bin:
        logger.warning("Alembic CLI not found on PATH, skipping auto-migration")
        return

    result = subprocess.run(
        [alembic_bin, "upgrade", "head"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Alembic migration failed: %s", result.stderr)
        raise RuntimeError(f"Database migration failed: {result.stderr}")
    logger.info("Database migrations applied successfully")


async def _fix_orphaned_sessions():
    """Reassign sessions with unknown agent_ids to the first matching user.

    This handles sessions created before the auth fix where agent_id
    was a random UUID instead of the logged-in user's ID.
    """
    from sqlalchemy import select, update
    from app.models import Session
    from app.models.user import User

    async with async_session_factory() as db:
        # Get all valid user IDs
        user_result = await db.execute(select(User.id))
        valid_user_ids = set(row[0] for row in user_result.all())

        if not valid_user_ids:
            return

        # Find sessions with agent_ids that don't match any user
        session_result = await db.execute(select(Session.id, Session.agent_id))
        orphaned = [
            (sid, aid) for sid, aid in session_result.all() if aid not in valid_user_ids
        ]

        if not orphaned:
            return

        # Get the first agent user (prefer agent role over admin)
        agent_user = await db.execute(
            select(User.id)
            .where(User.role == "agent", User.is_active.is_(True))
            .limit(1)
        )
        default_agent = agent_user.scalar_one_or_none()

        if not default_agent:
            # Fallback to any user
            any_user = await db.execute(
                select(User.id).where(User.is_active.is_(True)).limit(1)
            )
            default_agent = any_user.scalar_one_or_none()

        if not default_agent:
            return

        # Reassign orphaned sessions
        orphaned_ids = [sid for sid, _ in orphaned]
        await db.execute(
            update(Session)
            .where(Session.id.in_(orphaned_ids))
            .values(agent_id=default_agent)
        )
        await db.commit()
        logger.info(
            "Reassigned %d orphaned sessions to user %s",
            len(orphaned_ids),
            default_agent,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: run migrations and seed default data on startup."""
    # Import all models so Base.metadata knows about them
    import app.models  # noqa: F401

    # Run database migrations before any DB operations
    try:
        _run_migrations()
    except Exception as e:
        logger.error("Failed to run database migrations: %s", e, exc_info=True)
        raise

    try:
        # Seed default scenarios
        from app.services.seed_scenarios import seed_default_scenarios

        async with async_session_factory() as db:
            await seed_default_scenarios(db)

        # Seed default users
        from app.services.seed_users import seed_default_users

        async with async_session_factory() as db:
            await seed_default_users(db)

        # Seed demo dashboard data (agents, sessions, evaluations)
        from app.services.seed_demo_data import seed_demo_data

        async with async_session_factory() as db:
            await seed_demo_data(db)

        # Fix orphaned sessions: assign sessions with unknown agent_ids to existing users
        await _fix_orphaned_sessions()

        logger.info("Startup seeding complete")

    except Exception as e:
        logger.error("Startup error during seed: %s", e, exc_info=True)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Collection Agent Trainer",
        description="AI-powered training platform for collection agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Parse CORS origins from comma-separated string or "*"
    origins = (
        settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    )
    origins = [o.strip() for o in origins if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(campaigns.router, prefix="/api/campaigns", tags=["campaigns"])
    app.include_router(
        campaign_dashboard.router, prefix="/api/campaigns", tags=["campaign-dashboard"]
    )
    app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(voice.router, tags=["voice"])
    app.include_router(tts.router, prefix="/api", tags=["tts"])
    app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(config.router, prefix="/api", tags=["config"])

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/api/debug/sessions")
    async def debug_sessions(db=Depends(get_session)):
        """Debug endpoint: shows all sessions with their agent_ids and matching users."""
        from sqlalchemy import select
        from app.models import Session
        from app.models.user import User

        # Get all users
        users_result = await db.execute(select(User))
        users = {str(u.id): u.email for u in users_result.scalars().all()}

        # Get all sessions
        sessions_result = await db.execute(
            select(Session).order_by(Session.created_at.desc()).limit(20)
        )
        sessions_list = sessions_result.scalars().all()

        return {
            "users": users,
            "sessions": [
                {
                    "id": str(s.id),
                    "agent_id": str(s.agent_id),
                    "agent_email": users.get(
                        str(s.agent_id), "UNKNOWN - no matching user"
                    ),
                    "status": s.status,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in sessions_list
            ],
        }

    return app


app = create_app()
