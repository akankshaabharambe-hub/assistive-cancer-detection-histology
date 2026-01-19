# Minimal container for FastAPI inference service
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal). Pillow sometimes benefits from these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repo
COPY . /app

# Expose API port
EXPOSE 8000

# Start server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
