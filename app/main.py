from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import scenarios, sessions, voice, tts


def create_app() -> FastAPI:
    app = FastAPI(
        title="Collection Agent Trainer",
        description="AI-powered training platform for collection agents",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(voice.router, tags=["voice"])
    app.include_router(tts.router, prefix="/api", tags=["tts"])

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


app = create_app()
