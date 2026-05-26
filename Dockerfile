FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and tests
COPY src/ ./src
COPY tests/ ./tests

# Create directory for persistent local storage
RUN mkdir -p /app/storage

# Expose API and ZeroMQ ports
EXPOSE 8000
EXPOSE 5555

# Default command runs the FastAPI server
CMD ["python", "-m", "uvicorn", "src.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
