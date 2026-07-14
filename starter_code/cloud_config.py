import os
from collections.abc import Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator


class AppConfig(BaseModel):
    openai_model: str
    openai_embeddings_model: str
    litellm_api_key: str
    openai_base_url: str
    qdrant_url: str
    qdrant_api_key: str
    redis_url: str
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str

    @field_validator(
        "openai_model",
        "openai_embeddings_model",
        "litellm_api_key",
        "qdrant_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
    )
    @classmethod
    def validate_required_setting(cls, value: str) -> str:
        if not value or value == "<>":
            raise ValueError("must be configured")
        return value

    @field_validator("openai_base_url", "qdrant_url", "langfuse_host")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        parsed_url = urlparse(value)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("must be a valid HTTP(S) URL")
        return value

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        parsed_url = urlparse(value)
        if parsed_url.scheme not in {"redis", "rediss"} or not parsed_url.netloc:
            raise ValueError("must be a valid Redis URL")
        return value


def load_app_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    environment = os.environ if environ is None else environ
    litellm_api_key = environment.get("LITELLM_API_KEY") or environment.get(
        "OPENAI_API_KEY"
    )

    return AppConfig(
        openai_model=environment.get("OPENAI_MODEL", ""),
        openai_embeddings_model=environment.get("OPENAI_EMBEDDINGS_MODEL", ""),
        litellm_api_key=litellm_api_key or "",
        openai_base_url=environment.get("OPENAI_BASE_URL", ""),
        qdrant_url=environment.get("QDRANT_URL", ""),
        qdrant_api_key=environment.get("QDRANT_API_KEY", ""),
        redis_url=environment.get("REDIS_URL", ""),
        langfuse_host=(
            environment.get("LANGFUSE_HOST")
            or environment.get("LANGFUSE_BASE_URL")
            or ""
        ),
        langfuse_public_key=environment.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=environment.get("LANGFUSE_SECRET_KEY", ""),
    )
