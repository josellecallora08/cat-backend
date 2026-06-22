import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import scenarios, sessions, voice, tts, dashboard, auth
from app.config import settings
from app.database import Base, async_session_factory

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: seed default data on startup."""
    # Import all models so Base.metadata knows about them
    import app.models  # noqa: F401

    try:
        # Seed default scenarios
        from app.services.seed_scenarios import seed_default_scenarios

        async with async_session_factory() as db:
            await seed_default_scenarios(db)

        # Seed default users
        from app.services.seed_users import seed_default_users

        async with async_session_factory() as db:
            await seed_default_users(db)

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
    origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    origins = [o.strip() for o in origins if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(voice.router, tags=["voice"])
    app.include_router(tts.router, prefix="/api", tags=["tts"])
    app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


app = create_app()
