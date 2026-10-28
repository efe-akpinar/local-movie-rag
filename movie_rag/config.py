"""Runtime configuration. Every field is read from the environment variable of the same name."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # Index and retrieval (same models and sizes as the notebook)
    index_dir: Path = Path("artifacts/index")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieve_k: int = 20
    rerank_n: int = 5
    embed_batch_size: int = 128
    cross_encoder_batch_size: int = 32
    device: str = "auto"  # "auto" picks cuda when available, else cpu

    # Generation
    llm_provider: Literal["groq", "local", "fake"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_s: float = 20.0
    local_model: str = "Qwen/Qwen2.5-3B-Instruct"
    local_cpu_fallback_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 180

    # HTTP API
    max_question_chars: int = 300
    rate_limit_per_minute: int = 20  # per client IP and worker; 0 disables the limit
    cors_origins: str = ""  # comma-separated list of allowed origins; empty allows none
    # True only behind a proxy that appends the caller's address to X-Forwarded-For (Cloud Run, set in
    # deploy.yml); see client_ip() in api.py. False keys the rate limit on the socket address.
    trust_x_forwarded_for: bool = False
    metrics_public: bool = False  # serve GET /metrics; when false it returns 404 (on in docker compose)
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
