# PRD-262 US-007: Node.js 24 (LTS until April 2028) comes from the official image.
# It gets its own stage so the tag@digest pin sits on a FROM line, which is the
# only place an updater reads (it never reads a bare `COPY --from=<image>`).
FROM node:24-trixie-slim@sha256:7f486cabe892bff8fef04c0f7da8dba84a560be3e4e2740b7a86afc57ebec315 AS node

# Railway builds no target, so the last stage is what it ships: keep it last.
FROM python:3.12-slim AS production

WORKDIR /app

# System dependencies — DevOps toolchain for agent workspaces
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget openssh-client ca-certificates \
    build-essential \
    jq tree gosu \
    && rm -rf /var/lib/apt/lists/*

# Node.js 24 with npm and corepack, from the `node` stage: no NodeSource, so no
# extra apt updates, Debian python3.13 or gnupg. Not /usr/local/include: node-gyp
# downloads the headers on its first native build. The binary's libraries beyond
# glibc (libstdc++6, libgcc-s1) come with build-essential above.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY --from=node /usr/local/lib/node_modules/corepack /usr/local/lib/node_modules/corepack
# pnpm pinned exactly (PRD-262 §9-Q9: the current 12.x), and the npm cache the
# install leaves in /root/.npm cleaned in the same layer.
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && npm install -g --no-fund --no-audit pnpm@12.14.0 \
    && npm cache clean --force

# The Claude Code CLI is NOT installed here: claude-agent-sdk ships its own copy
# (site-packages/claude_agent_sdk/_bundled/claude) and resolves it FIRST —
# _find_cli() tries the bundled binary before shutil.which("claude"), so the npm
# global was 232 MB the SDK could never reach (PRD-233 slim pass). Node/npm/pnpm
# above stay: they are the agent workspace toolchain, not the CLI's runtime.

# uv only — a single fast installer an agent can use to pull whatever a task
# actually needs. Pre-installing pytest/ruff/black/mypy/isort/flake8/coverage/tox
# cost 185 MB in every install for tools most sessions never touch (PRD-233 slim pass).
RUN pip install --no-cache-dir uv

# Worker application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code (includes automatos_logging.py + automatos_metrics.py for observability)
COPY . .

# Non-root user (created at build time; entrypoint drops to this user)
RUN useradd -m -u 1000 worker \
    && mkdir -p /workspaces \
    && chown -R worker:worker /app /workspaces

# Headless Chromium for the workspace_html_to_png tool. THE DEFAULT IS THE
# PRODUCTION VALUE (true): Railway builds THIS file (railway.json: builder
# DOCKERFILE), so a `false` default here removed the browser from the prod
# worker on 2026-08-30 (#650). The local edition passes INSTALL_BROWSER=false
# explicitly via docker-compose.yml build args (Chromium + its X11/font stack
# is 1.18 GB for one tool); without it the tool fails with Playwright's own
# "Executable doesn't exist" message naming the install command — honest, not
# silent. Want it locally: docker compose build --build-arg INSTALL_BROWSER=true workspace-worker
# (Install deps manually: --with-deps fails on Debian Trixie due to renamed font pkgs.)
ARG INSTALL_BROWSER=true
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN if [ "$INSTALL_BROWSER" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2t64 libxshmfence1 \
        fonts-liberation fonts-noto-color-emoji fonts-unifont \
        && rm -rf /var/lib/apt/lists/* \
        && PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers playwright install chromium \
        && chmod -R o+rx /opt/playwright-browsers \
        && mkdir -p /home/worker/.cache && ln -sf /opt/playwright-browsers /home/worker/.cache/ms-playwright \
        && mkdir -p /root/.cache && ln -sf /opt/playwright-browsers /root/.cache/ms-playwright; \
    else \
        echo "Skipping Chromium (INSTALL_BROWSER=false) — workspace_html_to_png is unavailable"; \
    fi
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
