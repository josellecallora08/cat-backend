from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cat_db"
    debug: bool = False
    # CORS
    cors_origins: str = "*"

    # JWT Authentication
    jwt_secret: str = "change-this-to-a-random-secret-in-production"
    jwt_expiry_hours: int = 24
    reset_token_expiry_minutes: int = 30

    # Email (SMTP) for password reset
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_from_name: str = "CATS"
    smtp_use_tls: bool = True

    # Frontend URL for reset links
    frontend_url: str = "http://localhost:3000"

    # LLM configuration (Ollama/vLLM/Groq OpenAI-compatible API)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:32b"
    llm_api_key: str = ""
    llm_timeout: float = 30.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Lark (Feishu) OAuth
    lark_app_id: str = ""
    lark_app_secret: str = ""
    lark_redirect_uri: str = "http://localhost:3000/auth/lark/callback"

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:3000/auth/google/callback"

    # ElevenLabs TTS (leave empty to use gTTS fallback)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = (
        "pFZP5JQG7iQjIQuC4Bku"  # Lily - good for Filipino/multilingual
    )
    # TTS provider: "elevenlabs", "gtts", or "auto" (tries elevenlabs first)
    tts_provider: str = "auto"

    # Upload Security Configuration
    upload_max_file_size: int = 10_485_760  # 10 MB in bytes
    upload_accepted_extensions: str = ".pdf,.docx,.txt,.csv,.md"
    upload_quarantine_path: str = "./quarantine"
    upload_quarantine_retention_hours: int = 24
    upload_max_docx_entries: int = 500
    upload_max_docx_uncompressed_size: int = 52_428_800  # 50 MB
    upload_max_docx_depth: int = 2
    upload_rejection_max_attempts: int = 10
    upload_rejection_window_minutes: int = 60
    upload_rejection_cooldown_minutes: int = 30
    upload_scanner_enabled: bool = True
    upload_scanner_socket: str = "/var/run/clamav/clamd.ctl"
    upload_quarantine_cleanup_interval_minutes: int = 60

    # Script Registry Configuration
    script_max_definition_size_bytes: int = 262_144  # 256 KB in bytes
    script_max_trigger_phrases: int = 50
    script_max_expected_replies: int = 20
    script_max_escalation_conditions: int = 20
    script_max_field_text_length: int = 2000

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
