# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Annoy, pulled in by NeMo Guardrails, is built from source for Python 3.13.
RUN apt-get update && \
    apt-get install --no-install-recommends --yes g++ && \
    rm -rf /var/lib/apt/lists/*

# Keep dependency installation cached until project metadata changes.
COPY starter_code/pyproject.toml starter_code/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY starter_code/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.13-slim-bookworm AS runtime

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && \
    useradd --system --gid app --home-dir /app --no-create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/main.py /app/cloud_config.py /app/cloud_services.py ./
COPY --from=builder --chown=app:app /app/config ./config
COPY --from=builder --chown=app:app /app/datasets ./datasets

USER app

EXPOSE 8000

CMD ["gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--preload", "main:app"]
