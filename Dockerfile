# ─────────────────────────────────────────────────────────────────────────────
# Smart Parking Lot — Hugging Face Space Dockerfile
# Base: python:3.10-slim  |  Exposes: 7860 (Gradio)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── system deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── copy & install Python deps first (layer-cache friendly) ───────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy project source ───────────────────────────────────────────────────────
COPY . .

# ── create a non-root user (HF Spaces best practice) ─────────────────────────
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Gradio port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONPATH=/app
# ── entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "server/app.py"]
