# =============================================================================
# Automatos AI - Backend Dockerfile (Multi-stage)
# =============================================================================
# Stages:
#   - base: Common dependencies
#   - development: Hot-reload, debugging tools
#   - production: Optimized, minimal image
# =============================================================================

# =============================================================================
# Base Stage - Common Dependencies
# =============================================================================
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    libmagic1 \
    tesseract-ocr \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (works when Root Directory is set to automatos-ai)
COPY orchestrator/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Development Stage - Hot Reload & Debugging
# =============================================================================
FROM base as development

# Create minimal entrypoint script (Railway doesn't need full entrypoint - database is ready)
# For docker-compose, the full entrypoint.sh will be mounted/overridden
RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application code
COPY orchestrator/ .

# Create necessary directories
RUN mkdir -p /app/logs /app/vector_stores /app/projects

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command - can be overridden in docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production Stage - Optimized & Minimal
# =============================================================================
FROM base as production

# Copy application code only (no dev tools)
COPY orchestrator/ .

# Create minimal entrypoint script (Railway doesn't need full entrypoint - database is ready)
# For docker-compose, the full entrypoint.sh will be mounted/overridden
RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/vector_stores /app/projects

# Remove dev dependencies
RUN pip uninstall -y pytest pytest-asyncio black isort

# Create non-root user for security
RUN useradd -m -u 1000 automatos && \
    chown -R automatos:automatos /app

USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Production command - use PORT env var (Railway provides this)
# Default to 8000 if PORT not set
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"

