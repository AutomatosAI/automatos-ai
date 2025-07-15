
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    openssh-client \
    postgresql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/vector_stores /app/projects

# Create non-root user
RUN useradd --create-home --shell /bin/bash orchestrator && \
    chown -R orchestrator:orchestrator /app

# Switch to non-root user
USER orchestrator

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "orchestrator.py", "--help"]
