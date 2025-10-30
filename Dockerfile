# ============================
# 🏗️  Stage 1 — Builder Image
# ============================
FROM python:3.11-slim AS builder

# Set work directory inside container
WORKDIR /app

# Install system dependencies required for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies in a virtual environment
COPY requirements.txt .

# Handle potential UTF-16 encoding (convert to UTF-8 if needed)
RUN file -i requirements.txt | grep utf-16 && \
    iconv -f utf-16 -t utf-8 requirements.txt > req.txt && mv req.txt requirements.txt || true

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# =============================
# 🚀  Stage 2 — Runtime Image
# =============================
FROM python:3.11-slim

# Create a non-root user
RUN useradd -m app

# Set working directory
WORKDIR /app

# Copy virtual environment and project files from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY monitoring ./monitoring
COPY data ./data
COPY Makefile .
COPY dvc.yaml .
COPY infra ./infra

# Ensure virtualenv is used for all python/pip commands
ENV PATH="/opt/venv/bin:$PATH"

# Expose FastAPI default port
EXPOSE 8000

# Set ownership to non-root user
RUN chown -R app:app /app
USER app

# Healthcheck: periodically ping /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
