# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.14.1

# ---- builder: virtualenv with CPU-only PyTorch and the service package
FROM python:${PYTHON_VERSION}-slim AS builder
ARG TORCH_VERSION
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
WORKDIR /src
COPY pyproject.toml ./
COPY movie_rag ./movie_rag
RUN pip install .

# ---- models: bake the bi-encoder and cross-encoder in, so cold starts download nothing
FROM builder AS models
ENV HF_HOME=/opt/hf
COPY scripts/download_models.py ./scripts/
RUN python scripts/download_models.py

# ---- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    INDEX_DIR=/app/index \
    PORT=8080 \
    WEB_CONCURRENCY=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
COPY --from=builder /opt/venv /opt/venv
COPY --from=models --chown=app:app /opt/hf /opt/hf
WORKDIR /app
COPY gunicorn.conf.py ./
# Index artifacts come from the build context (scripts/build_index.py output; downloaded from GCS in CI).
COPY artifacts/index/ ./index/
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8080'), timeout=4)" || exit 1
CMD ["gunicorn", "--config", "gunicorn.conf.py", "movie_rag.api:app"]
