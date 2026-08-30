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
# -----------------------------------------------------------------------------
# Wheel-build stage — compilers live HERE and never reach the runtime image
# -----------------------------------------------------------------------------
# gcc/g++/dev headers are needed to build wheels, not to run them; leaving them
# in the shipped image cost ~250 MB in every install (PRD-233 slim pass).
# Railway builds with Railpack, not this file — SaaS is unaffected either way.
FROM python:3.11-slim as pybuild

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Knowledge-Graph clustering quality, opt-in (PRD-233 slim pass). The `[leiden]`
# extra on graphifyy pulls graspologic, and with it POT + llvmlite + numba +
# statsmodels + gensim + scikit-learn + matplotlib — ~660 MB — for ONE feature.
# graphify.cluster documents its own fallback: "Tries Leiden (graspologic)
# first… falls back to Louvain (built into networkx) if graspologic is not
# installed", so the Knowledge Graph still clusters without it, at lower
# partition quality.
# THE DEFAULT IS THE PRODUCTION VALUE (true). Railway builds THIS file
# (railway.json: builder DOCKERFILE, target production), so a Dockerfile default
# is a production decision — a `false` here shipped degraded clustering to prod
# on 2026-08-30 (#650). The local edition passes INSTALL_GRAPH_EXTRAS=false
# explicitly via docker-compose.yml build args to stay slim; want Leiden
# locally: docker compose build --build-arg INSTALL_GRAPH_EXTRAS=true backend
ARG INSTALL_GRAPH_EXTRAS=true

# futureagi is installed separately with --no-deps because it pins exact
# versions of common packages (requests==2.32.3, pandas==2.2.2, etc.)
# that conflict with our stack. Its code works fine with our versions.
# --prefix keeps the whole install in one tree the runtime stage copies.
RUN if [ "$INSTALL_GRAPH_EXTRAS" = "true" ]; then \
        cp requirements.txt requirements.build.txt; \
    else \
        sed 's/^graphifyy\[leiden\]/graphifyy/' requirements.txt > requirements.build.txt; \
    fi && \
    pip install --no-cache-dir --prefix=/install -r requirements.build.txt && \
    pip install --no-cache-dir --prefix=/install --no-deps futureagi==0.6.0 && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# -----------------------------------------------------------------------------
# Base — RUNTIME dependencies only
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

WORKDIR /app

# Runtime-only system deps. git: repo tooling; postgresql-client: the compose
# entrypoint's psql checks; libmagic1: python-magic; tesseract/ghostscript: OCR
# and PDF; pango/cairo/gdk-pixbuf: WeasyPrint (PRD-63). No compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    postgresql-client \
    libmagic1 \
    tesseract-ocr \
    ghostscript \
    libpango-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=pybuild /install /usr/local

# (NLTK data download removed: `nltk` is imported nowhere in the tree — 52 MB
# of corpora plus the package for dead weight. PRD-233 slim pass.)

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

# PRD-209: persisted credential-encryption key dir (compose mounts the backend_data
# volume here; ownership set BEFORE the USER switch so a fresh named volume inherits it).
RUN mkdir -p /app/data && chown -R automatos:automatos /app/data
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
# --timeout-graceful-shutdown (dev stage only; prod runs --workers below): a
# --reload first waits for open HTTP connections to close and the browser's SSE
# streams (notifications, board events) never do, so every code edit wedged the
# server in "Waiting for connections to close" until a restart (PRD-233 finding).
CMD sh -c "alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --port 8000 --reload --timeout-graceful-shutdown 5"

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

# PRD-209: persisted credential-encryption key dir (compose mounts the backend_data
# volume here; ownership set BEFORE the USER switch so a fresh named volume inherits it).
RUN mkdir -p /app/data && chown -R automatos:automatos /app/data
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
