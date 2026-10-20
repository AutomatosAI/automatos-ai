# syntax=docker/dockerfile:1
# PRD-262 US-045: the syntax line builds this file with the current 1.x Dockerfile
# frontend, which knows HEALTHCHECK --start-interval (below). Docker Engine 24's
# built-in frontend (BuildKit 0.11) rejects that flag.
# PRD-262 US-007: Node.js 24 (LTS until April 2028) comes from the official image.
# It gets its own stage so the tag@digest pin sits on a FROM line, which is the
# only place an updater reads (it never reads a bare `COPY --from=<image>`).
FROM node:24-trixie-slim@sha256:7f486cabe892bff8fef04c0f7da8dba84a560be3e4e2740b7a86afc57ebec315 AS node

# PRD-262 US-044: uv from its own image, in a named stage for the same reason.
FROM ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1 AS uv

# PRD-262 US-044: requirements.lock in two parts, claude-agent-sdk alone and
# everything else (the SDK's own dependencies included). The production stage
# copies each part by its content, so an SDK bump that leaves the rest of the
# lock alone re-runs only the SDK's install: from the build cache, the stable
# layer keeps its digest. awk keeps or drops a whole entry: the pin line, and
# the indented hash and `# via` lines under it. The build fails unless the SDK
# part holds the SDK and nothing else.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS locks
WORKDIR /locks
COPY requirements.lock .
RUN awk '/^[A-Za-z0-9]/ { sdk = /^claude-agent-sdk==/ } !sdk' requirements.lock > stable.lock \
    && awk '/^[A-Za-z0-9]/ { sdk = /^claude-agent-sdk==/ } sdk' requirements.lock > sdk.lock \
    && test "$(grep -c '^[A-Za-z0-9]' sdk.lock)" = 1 \
    && grep -q '^claude-agent-sdk==' sdk.lock \
    && ! grep -q '^claude-agent-sdk==' stable.lock

# Railway builds no target, so the last stage is what it ships: keep it last.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS production

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
# It is copied from the pinned uv image (PRD-262 US-044) to the path agents have
# always used, and every install below runs through it.
COPY --from=uv /uv /uvx /usr/local/bin/

# Worker application dependencies, from the lock (PRD-262 US-044), in three
# layers: everything but the SDK here, Playwright in the browser step, and the
# SDK last. Each install:
# - --require-hashes: every file is checked against the lock's sha256;
# - --no-deps: the lock is the whole resolution, so nothing else is fetched;
# - --system: into the image's own Python, where pip installed before;
# - --compile-bytecode: the .pyc are written now, as pip did, because the worker
#   runs as uid 1000 and cannot write them into site-packages. uv compiles all
#   of site-packages, but a later install finds the earlier layers' .pyc current
#   (timestamp-checked: no SOURCE_DATE_EPOCH here) and does not rewrite them;
# - --no-cache: uv's download cache stays out of the layer.
# No BuildKit cache mount: Railway fails a build whose cache mount lacks a
# hard-coded id=s/<service-id>-<path>, and the worker's service id is not known
# yet (PRD-262 §10). With it, each install becomes
# RUN --mount=type=cache,id=s/<worker-service-id>-/root/.cache/uv,target=/root/.cache/uv
# without --no-cache.
COPY --from=locks /locks/stable.lock /opt/worker-locks/
RUN uv pip install --system --no-cache --compile-bytecode --require-hashes --no-deps \
    -r /opt/worker-locks/stable.lock

# The worker user first, so everything after it can be created worker-owned:
# /app (the code lands below with --chown), the volume mount point and HOME's
# cache directory. The browser step runs as root, and before this order it made
# /home/worker/.cache root-owned on Railway only.
RUN useradd -m -u 1000 worker \
    && install -d -o worker -g worker /app /workspaces /home/worker/.cache

# Headless Chromium for the workspace_html_to_png tool. THE DEFAULT IS THE
# PRODUCTION VALUE (true): Railway builds THIS file (railway.json: builder
# DOCKERFILE), so a `false` default here removed the browser from the prod
# worker on 2026-08-30 (#650). The local edition passes INSTALL_BROWSER=false
# explicitly via docker-compose.yml build args, and so does the GHCR image
# (images.yml): no Playwright and no Chromium, so the tool reports Playwright
# missing. Want it locally: docker compose build --build-arg INSTALL_BROWSER=true workspace-worker
# The step sits above the code, so a code change never re-runs it (PRD-262 US-043):
# - Playwright comes from requirements-browser.lock (compiled against the main
#   lock, installed the same way), and its bundled Node (about 120 MB) is
#   deleted in the same RUN: the driver runs on the image's Node 24
#   (PLAYWRIGHT_NODEJS_PATH).
# - Only the headless shell: html_to_png launches headless with no channel
#   (executor.py), which runs the shell, never the full Chromium (about 600 MB).
# - The shell's libraries only, installed by hand (--with-deps fails on Debian
#   trixie's renamed font packages). Its DT_NEEDED list has no libcups, pango,
#   cairo or xshmfence. fontconfig is listed because without it the shell's
#   bundled fontconfig finds no /etc/fonts; fonts-unifont is the CJK fallback.
# - Mesa's GPU drivers (mesa-libgallium, LLVM, Z3: about 195 MB) are purged.
#   libgbm1 depends on them, but libgbm.so.1 links only libdrm and libexpat, and
#   the shell renders with its own SwiftShader. apt confirms first that libgbm1
#   is the only package left wanting them. That leaves apt with one unmet
#   dependency, so no apt-get install may follow this step.
# - The build fails if the shell or its bundled libraries miss a library.
ARG INSTALL_BROWSER=true
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers \
    PLAYWRIGHT_NODEJS_PATH=/usr/local/bin/node
COPY requirements-browser.lock .
RUN set -eu; \
    if [ "$INSTALL_BROWSER" != "true" ]; then \
        echo "Skipping the browser (INSTALL_BROWSER=$INSTALL_BROWSER): no Playwright, no Chromium; workspace_html_to_png is unavailable"; \
        exit 0; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        fontconfig libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64 \
        fonts-liberation fonts-noto-color-emoji fonts-unifont; \
    uv pip install --system --no-cache --compile-bytecode --require-hashes --no-deps -r requirements-browser.lock; \
    rm "$(python -c 'import os, playwright; print(os.path.dirname(playwright.__file__))')/driver/node"; \
    playwright install --only-shell chromium; \
    mesa="$(dpkg-query -W --showformat='${db:Status-Abbrev} ${Package}\n' \
        mesa-libgallium libllvm19 libz3-4 libxml2 libelf1t64 libsensors5 libsensors-config \
        libdrm-amdgpu1 libdrm-intel1 libpciaccess0 libx11-xcb1 libxshmfence1 \
        libxcb-dri3-0 libxcb-present0 libxcb-randr0 libxcb-sync1 libxcb-xfixes0 \
        2>/dev/null | awk '$1 == "ii" {print $2}')"; \
    plan="$(apt-get -s purge $mesa | awk '/^(Inst|Remv|Purg) / {sub(/:.*/, "", $2); print $1, $2}' | sort)"; \
    if [ "$plan" != "$(printf 'Purg %s\n' libgbm1 $mesa | sort)" ]; then \
        printf 'Purging Mesa would change more than libgbm1:\n%s\n' "$plan" >&2; exit 1; \
    fi; \
    dpkg --purge --force-depends $mesa; \
    rm -rf /var/lib/apt/lists/*; \
    shell="$(find "$PLAYWRIGHT_BROWSERS_PATH" -type f -name chrome-headless-shell)"; \
    test -x "$shell"; \
    if ldd "$shell" "$(dirname "$shell")"/*.so* | grep 'not found'; then \
        echo "chrome-headless-shell is missing the shared libraries listed above" >&2; exit 1; \
    fi; \
    chmod -R o+rx "$PLAYWRIGHT_BROWSERS_PATH"; \
    ln -s "$PLAYWRIGHT_BROWSERS_PATH" /home/worker/.cache/ms-playwright; \
    mkdir -p /root/.cache; \
    ln -s "$PLAYWRIGHT_BROWSERS_PATH" /root/.cache/ms-playwright

# claude-agent-sdk last among the installs, in a layer of its own (PRD-262 US-044):
# its dependencies are in the stable layer above, so a bump re-downloads this
# layer only, most of it the Claude CLI the wheel bundles.
COPY --from=locks /locks/sdk.lock /opt/worker-locks/
RUN uv pip install --system --no-cache --compile-bytecode --require-hashes --no-deps \
    -r /opt/worker-locks/sdk.lock

# Entrypoint fixes volume ownership then drops to worker user. --chmod sets the mode
# in the copy itself; a separate chmod RUN published an empty layer, since git already
# carries the exec bit (PRD-262 US-045).
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# Environment defaults
ENV WORKSPACE_VOLUME_PATH=/workspaces
ENV WORKSPACE_DEFAULT_QUOTA_GB=5
ENV WORKER_CONCURRENCY=3
ENV WORKER_HEALTH_PORT=8081
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app

# Worker code (includes automatos_logging.py + automatos_metrics.py for
# observability), last and worker-owned: a code change rebuilds only this layer.
COPY --chown=1000:1000 . .

# While the worker starts, Docker probes every 2 s instead of waiting a whole interval,
# so it reports healthy within seconds of the health server listening (PRD-262
# US-045). A compose healthcheck that sets no start_interval keeps this one: Docker
# fills each unset field from the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --start-interval=2s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

EXPOSE 8081

ENTRYPOINT ["entrypoint.sh"]
CMD []
