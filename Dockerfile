FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg нужен только для крайнего STT-фолбэка транскриптов
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY alembic.ini ./
COPY alembic ./alembic

# Сервису не нужен root: он ходит по сети и пишет в свой каталог логов.
# Права на каталог проставляются до объявления тома, чтобы том унаследовал их.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/logs/llm \
    && chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
