# =============================================================================
# Workspace Worker — PRD-56 Phase 2
# =============================================================================
# Consumes tasks from Redis queues, executes in isolated workspace directories.
# Pre-installs DevOps toolchain so agents don't need to pip/npm install basics.
#
# Build: docker build -f services/workspace-worker/Dockerfile -t automatos-workspace-worker .
# Run:   docker run -v workspace-data:/workspaces automatos-workspace-worker
# =============================================================================

FROM python:3.12-slim AS base

# Metadata
LABEL maintainer="Automatos AI <engineering@automatos.app>"
LABEL description="Workspace worker for agent task execution with physical workspaces"
LABEL prd="PRD-56 Phase 2"

# System dependencies — DevOps toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Version control & networking
    git curl wget openssh-client ca-certificates \
    # Build essentials (for native Python/Node packages)
    build-essential \
    # Useful CLI tools
    jq tree \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS (v20.x)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

# Python dev/test tools (pre-installed so agents can use immediately)
RUN pip install --no-cache-dir \
    pytest pytest-cov pytest-asyncio \
    ruff black mypy isort flake8 \
    coverage tox \
    uv

# Worker application dependencies
WORKDIR /app
COPY services/workspace-worker/requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r worker-requirements.txt

# Copy orchestrator (shared models, config, etc.)
COPY orchestrator/ /app/orchestrator/

# Copy worker service
COPY services/workspace-worker/ /app/worker/

# Non-root user
RUN useradd -m -u 1000 worker \
    && mkdir -p /workspaces \
    && chown -R worker:worker /app /workspaces

USER worker

# Workspace directory — attach a Railway Volume mounted at /workspaces
# (VOLUME instruction removed — Railway manages volumes externally)

# Environment defaults
ENV WORKSPACE_VOLUME_PATH=/workspaces
ENV WORKSPACE_DEFAULT_QUOTA_GB=5
ENV WORKER_CONCURRENCY=3
ENV WORKER_HEALTH_PORT=8081
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

EXPOSE 8081

# Entry point
CMD ["python", "-m", "worker.main"]
