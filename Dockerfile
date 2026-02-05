# Scanner - Advanced Deepfake Detection
# Production Dockerfile

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SCANNER_API_KEY=scanner-prod-key-change-me
ENV SCANNER_SECRET_KEY=change-this-to-a-secure-random-string

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash scanner
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-jose[cryptography] passlib[bcrypt]

# Copy application code
COPY --chown=scanner:scanner . .

# Download MediaPipe model if not present
RUN python -c "from preprocessing import FaceExtractor; FaceExtractor()" || true

# Create weights directory
RUN mkdir -p weights && chown -R scanner:scanner /app

# Switch to non-root user
USER scanner

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API server)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
