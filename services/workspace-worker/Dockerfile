# syntax=docker/dockerfile:1
# PRD-262 US-045: the syntax line builds this file with the current 1.x Dockerfile
# frontend, which knows HEALTHCHECK --start-interval (below). Docker Engine 24's
# built-in frontend (BuildKit 0.11) rejects that flag.
# PRD-262 US-007: Node.js 24 (LTS until April 2028) comes from the official image.
# It gets its own stage so the tag@digest pin sits on a FROM line, which is the
# only place an updater reads (it never reads a bare `COPY --from=<image>`).
FROM node:24-trixie-slim@sha256:7f486cabe892bff8fef04c0f7da8dba84a560be3e4e2740b7a86afc57ebec315 AS node

# Node, stripped. The same stage, byte for byte, in services/workspace-worker/Dockerfile
# and services/media-render/Dockerfile, so the two images share one node layer on a host.
FROM node AS node-stripped
RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && strip --strip-unneeded --preserve-dates /usr/local/bin/node \
 && rm -rf /var/lib/apt/lists/*

# npm without its docs and man pages (PRD-262, row 47: 2.67 MB). The image has no `man`,
# so `npm help` never showed them; `npm <command> --help` prints from npm's own code.
FROM node AS npm
RUN rm -rf /usr/local/lib/node_modules/npm/docs /usr/local/lib/node_modules/npm/man

# PRD-262 US-044: uv from its own image, in a named stage for the same reason.
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

# PRD-262 US-044: requirements.lock in three parts: claude-agent-sdk alone, psycopg2
# alone (the psycopg2 stage below builds it from source, PRD-262 row 27) and everything
# else (the SDK's own dependencies included). The production stage copies each part
# by its content, so an SDK bump that leaves the rest of the lock alone re-runs only
# the SDK's install: from the build cache, the stable layer keeps its digest. awk
# keeps or drops a whole entry: the pin line, and the indented hash and `# via` lines
# under it. The build fails unless the SDK and psycopg2 parts each hold their one
# package and nothing else, and the stable part holds neither.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS locks
WORKDIR /locks
COPY requirements.lock .
RUN awk '/^[A-Za-z0-9]/ { sdk = /^claude-agent-sdk==/; pg = /^psycopg2==/ } !sdk && !pg' requirements.lock > stable.lock \
    && awk '/^[A-Za-z0-9]/ { sdk = /^claude-agent-sdk==/ } sdk' requirements.lock > sdk.lock \
    && awk '/^[A-Za-z0-9]/ { pg = /^psycopg2==/ } pg' requirements.lock > psycopg2.lock \
    && test "$(grep -c '^[A-Za-z0-9]' sdk.lock)" = 1 \
    && grep -q '^claude-agent-sdk==' sdk.lock \
    && test "$(grep -c '^[A-Za-z0-9]' psycopg2.lock)" = 1 \
    && grep -q '^psycopg2==' psycopg2.lock \
    && ! grep -q '^claude-agent-sdk==' stable.lock \
    && ! grep -q '^psycopg2==' stable.lock

# psycopg2 compiled against Debian's libpq (PRD-262, row 27: 15.7 MB). psycopg2-binary's
# wheel carried its own libpq and two OpenSSLs, krb5, LDAP and SASL (16.1 MB), libraries
# Debian already installs in this image; the production stage installs libpq5 (0.4 MB)
# instead, so libpq gets the weekly OS rebuild's fixes too. Same project, version and
# API. The compiler and headers stay in this stage. The lock's psycopg2 entry is the
# hash-checked sdist, built in uv's isolated build environment and installed into
# /install, which the production stage copies whole; its extension's debug info and
# uv's lock file are left behind here.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS psycopg2
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=locks /locks/psycopg2.lock /locks/
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev libpq-dev binutils \
    && rm -rf /var/lib/apt/lists/* \
    && uv pip install --python /usr/local/bin/python --prefix /install --no-cache --compile-bytecode \
        --require-hashes --no-deps --no-binary psycopg2 -r /locks/psycopg2.lock \
    && find /install -type f -name '*.so*' -exec strip --strip-debug {} + \
    && rm -f /install/.lock

# Railway builds no target, so the last stage is what it ships: keep it last.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS production

# Node.js 24's binary, stripped of its symbol tables and debug sections (PRD-262, row 21:
# 20 MB). .dynsym stays, so native addons still resolve Node's N-API symbols
# (scripts/ci/image_worker_toolchain.py loads one). First, and with --link, so the layer
# is the same blob, on the same base, as media-render's: a host with both keeps one copy.
COPY --link --from=node-stripped /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# System dependencies — DevOps toolchain for agent workspaces. The same layer drops
# what no agent build or clone runs (scripts/ci/image_worker_toolchain.py proves the
# rest still works):
# - PRD-262 (row 16): gcc's lto-dump (29.4 MB on arm64). `gcc -flto` runs lto1,
#   lto-wrapper and the linker plugin, never lto-dump, which only prints LTO objects.
# - PRD-262 (row 19), the API's git trim (US-019): git's server-side and mail
#   binaries, both copies, which an agent workspace never serves. git-core's second
#   copy of git, which every builtin there links to, becomes a link to /usr/bin/git.
#   git-remote-http (clone, pull and push over HTTPS) and ssh stay.
# - PRD-262 (row 47): no gosu (2.9 MB). entrypoint.sh drops to the worker with
#   setpriv, which util-linux in the base image already carries.
# - PRD-262 (row 27): libpq5, the libpq the psycopg2 stage's build links. Its own
#   dependencies (libssl3t64, libgssapi-krb5-2, libldap2) are already here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget openssh-client ca-certificates \
    build-essential \
    jq tree libpq5 \
    && rm -f /usr/bin/*lto-dump* \
    && rm -f /usr/bin/git-shell /usr/bin/scalar \
        /usr/lib/git-core/git-daemon /usr/lib/git-core/git-http-backend \
        /usr/lib/git-core/git-http-push /usr/lib/git-core/git-imap-send \
        /usr/lib/git-core/git-shell /usr/lib/git-core/scalar \
    && ln -sf /usr/bin/git /usr/lib/git-core/git \
    && rm -rf /var/lib/apt/lists/*

# npm (from the `npm` stage, without its docs) and corepack for Node.js 24, whose binary
# is copied first, above. All from the official image: no NodeSource, so no extra apt
# updates, Debian python3.13 or gnupg. Not /usr/local/include: node-gyp downloads the
# headers on its first native build. The binary's libraries beyond glibc (libstdc++6,
# libgcc-s1) come with build-essential above.
COPY --from=npm /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY --from=node /usr/local/lib/node_modules/corepack /usr/local/lib/node_modules/corepack
# pnpm pinned exactly (PRD-262 §9-Q9: the current 12.x), and what the install leaves
# behind gone in the same layer: npm's cache in /root/.npm, its logs there, and Node 24's
# compile cache, which npm writes to /tmp/node-compile-cache (PRD-262, row 47: 1.4 MB).
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && npm install -g --no-fund --no-audit pnpm@12.14.0 \
    && npm cache clean --force \
    && rm -rf /tmp/node-compile-cache /root/.npm

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
# Then what runs, and nothing else (PRD-262, rows 40 and 47, the API's US-020 rule):
# strip --strip-debug drops the .debug_* sections from every compiled library the lock
# installed (8.1 MB, most of it cryptography's) and leaves .dynsym, which the loader
# resolves symbols through. It skips the auditwheel *.libs directories, libraries
# auditwheel rewrote with patchelf, the kind of file strip has broken before. Then every
# directory named exactly `tests` goes (0.62 MB: jsonschema's, referencing's and
# certifi's suites). scripts/ci/image_worker_site_packages.py loads every file the
# strip touched.
COPY --from=locks /locks/stable.lock /opt/worker-locks/
RUN uv pip install --system --no-cache --compile-bytecode --require-hashes --no-deps \
    -r /opt/worker-locks/stable.lock \
    && find /usr/local/lib/python3.12/site-packages -type f -name '*.so*' -not -path '*.libs/*' \
        -exec strip --strip-debug {} + \
    && find /usr/local/lib/python3.12/site-packages -type d -name tests -prune -exec rm -rf {} +

# psycopg2, built above against libpq5 (PRD-262, row 27), as a layer of its own.
COPY --from=psycopg2 /install /usr/local

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
