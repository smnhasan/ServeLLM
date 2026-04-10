from pydantic_settings import BaseSettings
from typing import List, Optional
import secrets


class Settings(BaseSettings):
    # App
    APP_NAME: str = "LLM Serve"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API
    API_V1_PREFIX: str = "/v1"

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    API_KEY_HEADER: str = "Authorization"
    # Comma-separated list of valid API keys (hashed or plain depending on AUTH_USE_HASHED)
    # For demo: plain keys. In production, store hashed.
    VALID_API_KEYS: str = "sk-llmserve-test-key-1234"
    AUTH_USE_HASHED: bool = False

    # Rate limiting (simple in-memory, per key per minute)
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_RPM: int = 60  # requests per minute per key

    # Models
    MODELS_CONFIG_PATH: str = "configs/models.yaml"
    DEFAULT_MODEL: str = "gpt-3.5-turbo"

    # Inference (dummy)
    DUMMY_MODE: bool = True  # Use dummy responses
    DUMMY_LATENCY_MS: int = 200  # Simulated latency

    # Streaming
    STREAM_CHUNK_DELAY_MS: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
