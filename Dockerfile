# ==============================================================================
# ZENOVA Multi-Stage Production Dockerfile
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Stage 2: Minimal Production Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ZENOVA_ENV=production \
    PATH="/opt/venv/bin:$PATH"

# Install minimal curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root system user and group
RUN groupadd -g 10001 zenova && \
    useradd -u 10001 -g zenova -s /bin/bash -m zenova

# Copy application source and configuration
COPY --chown=zenova:zenova src/ /app/src/
COPY --chown=zenova:zenova configs/ /app/configs/
COPY --chown=zenova:zenova models/ /app/models/
COPY --chown=zenova:zenova knowledge/ /app/knowledge/
COPY --chown=zenova:zenova migrations/ /app/migrations/
COPY --chown=zenova:zenova alembic.ini /app/alembic.ini
COPY --chown=zenova:zenova scripts/ /app/scripts/

# Create data and log directories with non-root ownership
RUN mkdir -p /app/data /app/backups /app/logs && \
    chown -R zenova:zenova /app

USER zenova:zenova

# Expose API port
EXPOSE 8000

# Container Healthcheck using liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Production server start
CMD ["uvicorn", "zenova.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--proxy-headers"]
