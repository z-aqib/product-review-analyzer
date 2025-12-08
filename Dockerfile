# Dockerfile
# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full source code and runtime assets
COPY src ./src
COPY artifacts ./artifacts
COPY data ./data
# needed for FAISS / embeddings, etc.

# Optional but nice: ensure Python can see /app
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
