# Dockerfile for Islamic AI Engine
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

WORKDIR /code

# Install system dependencies required for build & sqlite (for ChromaDB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python requirements
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and data
COPY . /code

# Create user for Hugging Face Spaces non-root execution
RUN useradd -m -u 1000 user && \
    chown -R user:user /code
USER user

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Run FastAPI app via uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
