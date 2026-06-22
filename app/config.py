from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cat_db"
    debug: bool = False
    # CORS
    cors_origins: str = "*"

    # JWT Authentication
    jwt_secret: str = "change-this-to-a-random-secret-in-production"
    jwt_expiry_hours: int = 24

    # LLM configuration (Ollama/vLLM/Groq OpenAI-compatible API)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:32b"
    llm_api_key: str = ""
    llm_timeout: float = 30.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # ElevenLabs TTS (leave empty to use gTTS fallback)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "pFZP5JQG7iQjIQuC4Bku"  # Lily - good for Filipino/multilingual
    # TTS provider: "elevenlabs", "gtts", or "auto" (tries elevenlabs first)
    tts_provider: str = "auto"

    model_config = {"env_prefix": "CAT_", "env_file": ".env"}

    @property
    def async_database_url(self) -> str:
        """Ensure the database URL uses the asyncpg driver."""
        url = self.database_url
        # Handle common non-async prefixes from hosting providers
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()
