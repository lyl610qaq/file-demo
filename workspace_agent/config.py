import os
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    workspace_root: Path = Path("workspace")
    seed_root: Path = Path("demo_workspace_seed")
    trace_root: Path = Path("traces")
    static_root: Path = Path("static")
    allowed_origin: str = "http://localhost:8000"
    max_model_calls: int = Field(30, ge=1, le=100)
    max_run_seconds: float = Field(300.0, ge=0.05, le=3600.0)
    max_concurrent_runs: int = Field(1, ge=1, le=8)
    max_read_bytes: int = Field(16384, ge=1024, le=65536)
    max_write_bytes: int = Field(262144, ge=1024, le=1048576)
    request_timeout_seconds: float = Field(60.0, ge=5.0, le=180.0)
    rate_limit_per_minute: int = Field(10, ge=1, le=120)
    trusted_proxy_cidrs: str = ""

    @model_validator(mode="after")
    def normalize_paths(self) -> Self:
        for field_name in (
            "workspace_root",
            "seed_root",
            "trace_root",
            "static_root",
        ):
            path = getattr(self, field_name)
            normalized = Path(os.path.abspath(path.expanduser()))
            setattr(self, field_name, normalized)
        return self
