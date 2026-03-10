"""Application settings managed via pydantic-settings."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr

    extraction_model: str = "anthropic:claude-haiku-4-5"
    categorization_model: str = "anthropic:claude-haiku-4-5"
    report_model: str = "anthropic:claude-haiku-4-5"
    ollama_base_url: str = "http://localhost:11434/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    settings = Settings()  # type: ignore[call-arg]
    # Export env vars so pydantic-ai providers can find them
    os.environ.setdefault(
        "ANTHROPIC_API_KEY", settings.anthropic_api_key.get_secret_value()
    )
    os.environ.setdefault("OLLAMA_BASE_URL", settings.ollama_base_url)
    return settings
