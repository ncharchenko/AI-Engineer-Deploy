import os
from collections.abc import Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    openai_model: str = Field(validation_alias="OPENAI_MODEL")
    openai_embeddings_model: str = Field(
        validation_alias="OPENAI_EMBEDDINGS_MODEL"
    )
    litellm_api_key: str = Field(validation_alias="LITELLM_API_KEY")
    openai_base_url: str = Field(validation_alias="OPENAI_BASE_URL")
    qdrant_url: str = Field(validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(validation_alias="QDRANT_API_KEY")
    redis_url: str = Field(validation_alias="REDIS_URL")
    langfuse_host: str = Field(validation_alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(validation_alias="LANGFUSE_SECRET_KEY")

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
    return AppConfig.model_validate(environment)
