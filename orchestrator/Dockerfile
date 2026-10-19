# =============================================================================
# Automatos AI - Backend Dockerfile (Multi-stage)
# =============================================================================
# Stages:
#   - base: Common dependencies
#   - development: Hot reload; no code (compose mounts the source over /app)
#   - compile: the code, with its boot modules compiled; never ships itself
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
# Railway builds this file too (its last stage, production; default args), so the
# slim pass applies to SaaS as well.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce as pybuild

WORKDIR /app

# binutils: the strip below (PRD-262 US-020). gcc brings it too; named so the strip
# keeps it if the compilers go.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    binutils \
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
# (railway.json: builder DOCKERFILE; the last stage), so a Dockerfile default
# is a production decision — a `false` here shipped degraded clustering to prod
# on 2026-08-30 (#650). The local edition passes INSTALL_GRAPH_EXTRAS=false
# explicitly via docker-compose.yml build args to stay slim; want Leiden
# locally: docker compose build --build-arg INSTALL_GRAPH_EXTRAS=true backend
ARG INSTALL_GRAPH_EXTRAS=true

# --prefix keeps the whole install in one tree the runtime stage copies.
# Wheels only (PRD-262 US-015): a package that would build from source fails the
# build instead. The two exceptions are pure-Python sdists: pysher (composio's) and
# graspologic (the [leiden] extra; it leaves the list with PRD-262 US-027). The order
# matters: `--only-binary :all:` clears every per-package exception named before it,
# so `--no-binary` comes after.
RUN if [ "$INSTALL_GRAPH_EXTRAS" = "true" ]; then \
        cp requirements.txt requirements.build.txt; \
    else \
        sed 's/^graphifyy\[leiden\]/graphifyy/' requirements.txt > requirements.build.txt; \
    fi && \
    pip install --no-cache-dir --prefix=/install --only-binary :all: --no-binary pysher,graspologic \
        -r requirements.build.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# What runs, and nothing else (PRD-262 US-020). strip --strip-debug drops the .debug_*
# sections from every compiled library (about 91 MB of the 16 Oct pip layer) and leaves
# .dynsym, which the loader resolves symbols through, as it is. It skips the auditwheel
# *.libs directories: they hold about 1 MB of it, in libraries auditwheel renamed and
# rewrote with patchelf, the kind of file strip has broken before. The extensions that
# link them were rewritten too, so CI loads every file the strip touched
# (scripts/ci/image_site_packages.py). Then every directory named exactly `tests` under
# site-packages goes (about 108 MB); numpy.testing, sqlalchemy.testing and
# alembic.testing are importable APIs, not test suites, and stay.
RUN find /install -type f -name '*.so*' -not -path '*.libs/*' -exec strip --strip-debug {} + && \
    find /install -type d -name tests -path '*/site-packages/*' -prune -exec rm -rf {} +

# botocore keeps the models of the services we call, and no others (PRD-262 US-021).
# botocore/data holds one for every AWS service, 437 directories and about 18.6 MB.
# The app builds s3, s3vectors and bedrock-runtime clients; botocore's credential
# providers build sts (assumed roles, web identity), sso and sso-oidc (an SSO profile).
# The top-level files (endpoints, partitions, retry and default configuration) stay.
# A kept service missing from the install fails the build before anything goes. Any
# other client raises UnknownServiceError: test_prd262_botocore_services.py fails when
# code in orchestrator/ names one, and CI builds each kept client inside the image
# (scripts/ci/image_botocore.py).
RUN keep="s3 s3vectors bedrock-runtime sts sso sso-oidc" && \
    data=/install/lib/python3.11/site-packages/botocore/data && \
    for service in $keep; do \
        test -d "$data/$service" || { echo "botocore/data has no $service" >&2; exit 1; }; \
    done && \
    for dir in "$data"/*/; do \
        name=$(basename "$dir"); \
        case " $keep " in *" $name "*) ;; *) rm -rf "$dir" || exit 1 ;; esac; \
    done

# -----------------------------------------------------------------------------
# Base — RUNTIME dependencies only
# -----------------------------------------------------------------------------
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce as base

WORKDIR /app

# Runtime-only system deps, each one something the code calls (PRD-262 US-019). No
# compilers.
# - git: repo tooling (the code graph clones through GitPython, the skill loader
#   runs the CLI); postgresql-client: the compose entrypoint's psql checks;
#   libmagic1: python-magic; tesseract-ocr: OCR (processors.py).
# - pango, pangoft2 and harfbuzz-subset: the libraries WeasyPrint opens (PRD-63),
#   named because only tesseract brought the last two. libcairo2: tesseract needs it.
# - fonts-dejavu-core: the ballot boxes (U+2610/U+2611) a printed checklist carries
#   (F298), named rather than left to whichever font fontconfig pulls in.
# - fonts-urw-base35: the faces fontconfig gives a brand kit that names Helvetica,
#   Times, Courier, Palatino or Bookman with no font file (Nimbus, C059, P052, all
#   metric-compatible), so its PDF keeps those metrics instead of DejaVu's. It came
#   with ghostscript, which nothing called.
# The same layer deletes tesseract's orientation model (the one OCR call uses the
# default page mode, which never loads it; tesseract-ocr depends on the package) and
# git's server-side and mail binaries, both copies. git-core's second copy of git,
# which every builtin there links to, becomes a link to /usr/bin/git.
# git-remote-http, which clones over HTTPS, stays.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    postgresql-client \
    libmagic1 \
    tesseract-ocr \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-subset0 \
    libcairo2 \
    fonts-dejavu-core \
    fonts-urw-base35 \
    && rm -f /usr/share/tesseract-ocr/5/tessdata/osd.traineddata \
    && rm -f /usr/bin/git-shell /usr/bin/scalar \
        /usr/lib/git-core/git-daemon /usr/lib/git-core/git-http-backend \
        /usr/lib/git-core/git-http-push /usr/lib/git-core/git-imap-send \
        /usr/lib/git-core/git-shell /usr/lib/git-core/scalar \
    && ln -sf /usr/bin/git /usr/lib/git-core/git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=pybuild /install /usr/local

# tiktoken's encodings, fetched here so the API counts tokens with no internet
# (PRD-262 US-033). tiktoken downloads an encoding it has no cached copy of, through
# requests.get with no timeout, into a temp directory no new container keeps; on a
# host whose egress drops packets that hangs for minutes. TIKTOKEN_CACHE_DIR is read
# by tiktoken itself, not by config.py (FR-12), and holds one hash-checked file per
# encoding, readable by the runtime user. `encodings` lists every encoding the code
# requests: core/context_guard.py's TOKENIZER_ENCODING, the only one today
# (orchestrator/tests/test_prd262_offline_boot.py fails when the code asks for one
# that is not here). The second load goes through a proxy that refuses every
# connection, so a cache that cannot serve them offline fails the build. -B: the layer
# holds the cache and nothing else (the stdlib's bytecode is the production stage's).
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN encodings="cl100k_base" && \
    load='import sys, tiktoken; [tiktoken.get_encoding(name) for name in sys.argv[1:]]' && \
    python -B -c "$load" $encodings && \
    HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 python -B -c "$load" $encodings

# =============================================================================
# Development Stage - Hot Reload & Debugging
# =============================================================================
FROM base as development

# No application code (PRD-262 US-017). Both compose files bind-mount ./orchestrator
# over /app, which hid a baked copy, while `COPY . .` and a `chown -R /app` stored the
# whole build context twice on every build, a checkout's uploads and caches included.
# Run this stage with the source mounted at /app.
#
# The user and the directories the app writes, owned by it (no -R: they are empty), so
# a fresh backend_logs or backend_data named volume mounted on one starts writable
# (PRD-209: the credential key lives in /app/data). The production stage's first RUN,
# word for word, so the two stages share the layer.
RUN useradd -m -u 1000 automatos && \
    mkdir -p /app/logs /app/vector_stores /app/projects /app/exports /app/data && \
    chown automatos:automatos /app /app/logs /app/vector_stores /app/projects /app/exports /app/data

# The wait-migrate-seed entrypoint, the only owner of migrations (PRD-262 US-029):
# before the server it runs the migration step, plus the seeds where
# AUTOMATOS_MIGRATE_ON_BOOT=true (compose sets it) asks. Compose also bind-mounts
# it, so edits apply without a rebuild.
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# The healthcheck (F416): the HEALTHCHECK below and the entrypoint's watchdog both run it.
COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh
USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD docker-healthcheck.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command — the hot-reload server. It runs no alembic: the entrypoint
# migrates, once, before it starts (PRD-262 US-029).
# --timeout-graceful-shutdown: a --reload first waits for open HTTP connections to
# close and the browser's SSE streams (notifications, board events) never do, so
# every code edit wedged the server in "Waiting for connections to close" until a
# restart (PRD-233 finding).
# `exec` (F416): uvicorn replaces the shell, so the entrypoint's watchdog and a
# `docker stop` signal uvicorn itself, not a shell that would die without it.
CMD exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload --timeout-graceful-shutdown 5

# =============================================================================
# Compile Stage - the code, and its boot modules' bytecode (PRD-262 US-032)
# =============================================================================
# The production image's only source of app bytecode: .dockerignore keeps a
# checkout's .pyc out of the context (PRD-262 US-017), and nothing else writes any
# under /app. FROM base, so the bytecode is the runtime interpreter's. It compiles
# the modules every API process imports at boot (scripts/boot_modules.py: main's
# static import closure, the routers router_manifest.py mounts and the entrypoint's
# seed check), not all of /app: bytecode changes with every code edit, so it rides
# in every code-only update. unchecked-hash: Python never compares a .pyc with its
# source, so the build's file times can't make one stale, and a new container
# imports the boot set instead of compiling it first. PYTHONHASHSEED=0 fixes the
# string hashes, the one input of a .pyc that changes from run to run. The list
# goes to /tmp, which the production stage does not copy. §9-Q7: the bytecode stays
# only if .github/workflows/bytecode-experiment.yml shows a new container reaching
# /health 200 at least 2 s sooner with it, within US-016's update bar.
FROM base as compile

COPY . .
RUN python scripts/boot_modules.py > /tmp/boot-modules.txt && \
    PYTHONHASHSEED=0 python -m compileall -q --invalidation-mode unchecked-hash -i /tmp/boot-modules.txt

# =============================================================================
# Production Stage - Optimized & Minimal
# =============================================================================
FROM base as production

# The runtime user first, then /app and the directories the app writes, owned by
# it (no -R: they are empty), so a fresh backend_logs or backend_data named volume
# mounted on one starts writable (PRD-209: the credential key lives in /app/data).
# The code arrives already owned by the user (the last COPY below), so no
# `chown -R /app` layer stores every file a second time (PRD-262 US-016).
# useradd gives the automatos group the user's id, 1000.
RUN useradd -m -u 1000 automatos && \
    mkdir -p /app/logs /app/vector_stores /app/projects /app/exports /app/data && \
    chown automatos:automatos /app /app/logs /app/vector_stores /app/projects /app/exports /app/data

# The stdlib's bytecode, compiled once as root (PRD-262 US-014): the python base
# ships none, and the automatos user can't write to /usr/local, so without it
# every process compiles each stdlib module it imports, every time. site-packages
# is excluded: its bytecode comes only from pybuild. unchecked-hash .pyc stay
# valid whatever the build clamps file mtimes to. Before the COPYs, so neither a
# code change nor an entrypoint edit rebuilds this layer.
RUN python -m compileall -q -x '/site-packages/' --invalidation-mode unchecked-hash /usr/local/lib/python3.11

# The wait-migrate-seed entrypoint, the only owner of migrations (PRD-262 US-029).
# Before the server it runs the migration step, so Railway (no flag set) migrates
# there; AUTOMATOS_MIGRATE_ON_BOOT=true adds the seeds (compose), and
# AUTOMATOS_SKIP_MIGRATIONS=true runs nothing (the Helm api pods).
# `docker-entrypoint.sh migrate` runs the full lifecycle and exits (the Kubernetes
# migration Job).
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# The healthcheck (F416): the HEALTHCHECK below and the entrypoint's watchdog both run it.
COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh

# The application code, with its boot modules' bytecode from the compile stage
# (PRD-262 US-032): the stage's last filesystem layer, code and bytecode together,
# so a code-only update downloads this one layer (PRD-262 US-016; images.yml
# judges it against .github/image-budgets.json's update bar). Owned by the user
# created above, by its numeric id. Everything after it only sets image metadata.
# The test tools are in requirements-dev.txt, which only CI installs;
# .dockerignore keeps the rest of the context out.
COPY --from=compile --chown=1000:1000 /app /app
USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD docker-healthcheck.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# The worker count, which uvicorn reads from WEB_CONCURRENCY when the command
# names no --workers (PRD-262 US-029). 4 is Railway's, which sets none;
# envs/api.defaults sets 1 for compose. Set it on a service to change it there.
ENV WEB_CONCURRENCY=4

# Production command — uvicorn, and no alembic: the entrypoint migrates, once,
# before it starts, and a failed migration exits before uvicorn runs (PRD-262
# US-029). Shell form, so the shell expands Railway's PORT. `exec` (F416):
# uvicorn replaces the shell, so `docker stop` and the entrypoint's watchdog
# signal uvicorn itself. No --timeout-graceful-shutdown here, so Railway's
# shutdown is unchanged; compose sets UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN
# (envs/api.defaults).
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
