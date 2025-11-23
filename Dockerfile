# ET Social Intelligence System - Dockerfile
# Production-ready container with all dependencies pre-installed

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models (baked into image for reproducibility)
RUN python -m spacy download en_core_web_lg || \
    python -m spacy download en_core_web_md || \
    python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p et_intel/data/uploads \
             et_intel/data/processed \
             et_intel/data/database \
             et_intel/reports/charts \
             et_intel/reports/pdfs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for Streamlit dashboard (if running)
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "-m", "et_intel.cli.cli", "--help"]

