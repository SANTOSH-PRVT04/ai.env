# ── QTrack AI Environment — Dockerfile ───────────────────────────────
# OpenEnv Hackathon | XYZ Hospital Team
# Python 3.10 slim base (matches HF Spaces sdk_version config)

FROM python:3.10-slim

# Metadata
LABEL maintainer="SANTOSH-PRVT04"
LABEL description="QTrack AI Environment — Hospital Queue Optimization"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY app.py .
COPY env.py .
COPY grader.py .
COPY openenv.yaml .
COPY README.md .

# Create non-root user for security
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose Gradio default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Launch the app
CMD ["python", "app.py"]
