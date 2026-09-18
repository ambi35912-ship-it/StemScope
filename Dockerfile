FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
RUN useradd --create-home --uid 10001 stemscope && mkdir /storage && chown stemscope /storage
ENV PATH="/app/.venv/bin:$PATH" STEMSCOPE_OUTPUT_DIR=/storage/outputs TORCH_HOME=/storage/torch \
    MPLCONFIGDIR=/storage/matplotlib GRADIO_ANALYTICS_ENABLED=False
USER stemscope
EXPOSE 8000
CMD ["python", "-m", "stemscope.container_api"]
