from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cat_db"
    debug: bool = False
    cors_origins: list[str] = ["*"]

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

    model_config = {"env_prefix": "CAT_", "env_file": ".env"}


settings = Settings()
