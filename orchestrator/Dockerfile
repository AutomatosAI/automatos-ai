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
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    postgresql-client \
    libmagic1 \
    tesseract-ocr \
    ghostscript \
    # WeasyPrint system deps (PRD-63: Document Generation)
    libpango-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements (Dockerfile is in orchestrator/, so copy from current dir)
COPY requirements.txt .

# Install Python dependencies (clean cache after)
# futureagi is installed separately with --no-deps because it pins exact
# versions of common packages (requests==2.32.3, pandas==2.2.2, etc.)
# that conflict with our stack. Its code works fine with our versions.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps futureagi==0.6.0 && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Download NLTK data (punkt tokenizer, stopwords) for memory operations
# Set NLTK_DATA to a system-wide location accessible to all users
ENV NLTK_DATA=/usr/local/nltk_data
RUN python -c "import nltk; nltk.download('punkt', quiet=True, download_dir='/usr/local/nltk_data'); nltk.download('stopwords', quiet=True, download_dir='/usr/local/nltk_data')" && \
    chmod -R 755 /usr/local/nltk_data && \
    rm -rf /root/.cache/nltk

# =============================================================================
# Development Stage - Hot Reload & Debugging
# =============================================================================
FROM base as development

# Create minimal entrypoint script (Railway doesn't need full entrypoint - database is ready)
# For docker-compose, the full entrypoint.sh will be mounted/overridden
RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application code (Dockerfile is in orchestrator/, so copy from current dir)
COPY . .

# Create necessary directories and non-root user (matches production stage)
RUN mkdir -p /app/logs /app/vector_stores /app/projects /app/exports && \
    useradd -m -u 1000 automatos && \
    chown -R automatos:automatos /app

USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command — apply pending alembic migrations before starting,
# so the deployed code and the live schema can never drift. ``heads``
# (plural) is required because the migration tree currently has
# multiple unmerged heads; ``head`` (singular) errors out. If any
# migration fails the container fails to boot — loud is better than
# silent UndefinedColumn errors at request time.
CMD sh -c "alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

# =============================================================================
# Production Stage - Optimized & Minimal
# =============================================================================
FROM base as production

# Copy application code only (no dev tools)
# .dockerignore will exclude unnecessary files
COPY . .

# Create minimal entrypoint script (Railway doesn't need full entrypoint - database is ready)
# For docker-compose, the full entrypoint.sh will be mounted/overridden
RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/vector_stores /app/projects /app/exports

# Remove dev dependencies and clean up
RUN pip uninstall -y pytest pytest-asyncio black isort 2>/dev/null || true && \
    pip cache purge && \
    rm -rf /root/.cache/pip /root/.cache/pytest /tmp/* && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true

# Create non-root user for security
RUN useradd -m -u 1000 automatos && \
    chown -R automatos:automatos /app

USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Production command — apply pending alembic migrations before starting,
# so deploys can't ship a model change ahead of the schema. ``heads``
# (plural) is required because the migration tree currently has multiple
# unmerged heads; ``head`` (singular) errors with "Multiple head
# revisions are present". ``&&`` ensures uvicorn never starts if a
# migration fails (visible deploy failure beats silent UndefinedColumn
# errors at request time). Concurrent replicas serialise via alembic's
# row lock — second runner sees no pending migrations and continues.
CMD sh -c "alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
