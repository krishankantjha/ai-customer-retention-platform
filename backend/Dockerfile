FROM python:3.10-slim

WORKDIR /app

# Create system group and non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Install system dependencies (build-essential needed for compiling any custom native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for caching
COPY --chown=appuser:appgroup backend/requirements.txt /app/backend/requirements.txt

# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy source code, ML pipelines, and log configs
COPY --chown=appuser:appgroup backend /app/backend
COPY --chown=appuser:appgroup ml /app/ml
COPY --chown=appuser:appgroup configs /app/configs

# Setup logs directory and adjust permissions
RUN mkdir -p /app/backend/logs && \
    chown -R appuser:appgroup /app/backend/logs

# Set Python path environment variable
ENV PYTHONPATH=/app

# Set working directory to backend folder
WORKDIR /app/backend

# Use the non-root system user
USER appuser

# Expose backend port
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

