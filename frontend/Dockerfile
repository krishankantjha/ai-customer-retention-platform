FROM python:3.10-slim

WORKDIR /app

# Create system group and non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy requirements file first for caching
COPY --chown=appuser:appgroup frontend/requirements.txt /app/frontend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/frontend/requirements.txt

# Copy frontend source code
COPY --chown=appuser:appgroup frontend /app/frontend

# Set working directory to frontend folder
WORKDIR /app/frontend

# Use the non-root system user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start Streamlit application
CMD ["streamlit", "run", "app.py"]

