FROM python:3.12-slim

WORKDIR /app

# System dependencies — DevOps toolchain for agent workspaces
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget openssh-client ca-certificates \
    build-essential \
    jq tree gosu \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS (v20.x)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI — required by claude-agent-sdk for headless canvas
# sessions (PRD-170 S1). The SDK spawns this binary per workspace session.
RUN npm install -g @anthropic-ai/claude-code

# Python dev/test tools (pre-installed so agents can use immediately)
RUN pip install --no-cache-dir \
    pytest pytest-cov pytest-asyncio \
    ruff black mypy isort flake8 \
    coverage tox \
    uv

# Worker application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code (includes automatos_logging.py + automatos_metrics.py for observability)
COPY . .

# Non-root user (created at build time; entrypoint drops to this user)
RUN useradd -m -u 1000 worker \
    && mkdir -p /workspaces \
    && chown -R worker:worker /app /workspaces

# Headless Chromium for workspace_html_to_png tool (PRD-social-render).
# Install deps manually (--with-deps fails on Debian Trixie due to renamed font pkgs).
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2t64 libxshmfence1 \
    fonts-liberation fonts-noto-color-emoji fonts-unifont \
    && rm -rf /var/lib/apt/lists/* \
    && PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers playwright install chromium \
    && chmod -R o+rx /opt/playwright-browsers \
    && mkdir -p /home/worker/.cache && ln -sf /opt/playwright-browsers /home/worker/.cache/ms-playwright \
    && mkdir -p /root/.cache && ln -sf /opt/playwright-browsers /root/.cache/ms-playwright

# Entrypoint fixes volume ownership then drops to worker user
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Environment defaults
ENV WORKSPACE_VOLUME_PATH=/workspaces
ENV WORKSPACE_DEFAULT_QUOTA_GB=5
ENV WORKER_CONCURRENCY=3
ENV WORKER_HEALTH_PORT=8081
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

EXPOSE 8081

ENTRYPOINT ["entrypoint.sh"]
CMD []
