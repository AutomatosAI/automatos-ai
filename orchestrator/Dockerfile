# syntax=docker/dockerfile:1
# =============================================================================
# Automatos AI - Backend Dockerfile (Multi-stage)
# =============================================================================
# Stages:
#   - base: Common dependencies
#   - development: Hot reload; no code (docker-compose.dev.yml, `make dev`, mounts
#     the source over /app)
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

# =============================================================================
# Development Stage - Hot Reload & Debugging
# =============================================================================
FROM base as development

# No application code (PRD-262 US-017). docker-compose.dev.yml (`make dev`) bind-mounts
# ./orchestrator over /app, which hid a baked copy, while `COPY . .` and a `chown -R /app`
# stored the whole build context twice on every build, a checkout's uploads and caches
# included. Run this stage with the source mounted at /app. The default compose stack
# runs the production stage below (PRD-262 US-035).
#
# The user and the directories the app writes, owned by it (no -R: they are empty), so
# a fresh backend_logs or backend_data named volume mounted on one starts writable
# (PRD-209: the credential key lives in /app/data). The production stage's first RUN,
# word for word, so the two stages share the layer.
RUN useradd -m -u 1000 automatos && \
    mkdir -p /app/logs /app/vector_stores /app/projects /app/exports /app/data && \
    chown automatos:automatos /app /app/logs /app/vector_stores /app/projects /app/exports /app/data

# The wait-migrate-seed entrypoint. It runs the lifecycle only for `migrate` or
# AUTOMATOS_MIGRATE_ON_BOOT=true (compose sets it); otherwise it runs the CMD
# directly. Compose also bind-mounts it, so edits apply without a rebuild.
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# The healthcheck (F416): the HEALTHCHECK below and the entrypoint's watchdog both run it.
COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh
# The placeholder-secret guard the entrypoint runs before a command (PRD-262 US-037).
COPY --chmod=0755 docker-secrets-guard.sh /usr/local/bin/docker-secrets-guard.sh
USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD docker-healthcheck.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command — apply pending alembic migrations before starting,
# so the deployed code and the live schema can never drift. ``heads``
# (plural) upgrades the repository's single migration head. The
# single-head invariant is enforced by the migration guard tests. If any
# migration fails the container fails to boot — loud is better than
# silent UndefinedColumn errors at request time.
# --timeout-graceful-shutdown (dev stage only; prod runs --workers below): a
# --reload first waits for open HTTP connections to close and the browser's SSE
# streams (notifications, board events) never do, so every code edit wedged the
# server in "Waiting for connections to close" until a restart (PRD-233 finding).
# `exec` (F416): uvicorn replaces the shell, so the entrypoint's watchdog and a
# `docker stop` signal uvicorn itself, not a shell that would die without it.
CMD sh -c "alembic upgrade heads && exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload --timeout-graceful-shutdown 5"

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

# The wait-migrate-seed entrypoint. By default it runs the CMD directly, so
# Railway boots exactly as before (the CMD migrates). `docker-entrypoint.sh
# migrate` runs the full lifecycle and exits (the Kubernetes migration Job);
# AUTOMATOS_MIGRATE_ON_BOOT=true runs it before the CMD.
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# The healthcheck (F416): the HEALTHCHECK below and the entrypoint's watchdog both run it.
COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh
# The placeholder-secret guard the entrypoint runs before a command (PRD-262 US-037).
COPY --chmod=0755 docker-secrets-guard.sh /usr/local/bin/docker-secrets-guard.sh

# The application code: the stage's last filesystem layer, so a code-only update
# downloads this one layer (PRD-262 US-016; images.yml judges it against
# .github/image-budgets.json's update bar). Owned by the user created above, by
# its numeric id. Everything after it only sets image metadata. The test tools
# are in requirements-dev.txt, which only CI installs; .dockerignore keeps the
# rest of the context out.
COPY --chown=1000:1000 . .
USER automatos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD docker-healthcheck.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Production command — apply pending alembic migrations before starting,
# so deploys can't ship a model change ahead of the schema. ``heads``
# (plural) upgrades the repository's single migration head. The
# single-head invariant is enforced by the migration guard tests.
# ``&&`` ensures uvicorn never starts if a
# migration fails (visible deploy failure beats silent UndefinedColumn
# errors at request time). Concurrent replicas serialise via alembic's
# row lock — second runner sees no pending migrations and continues.
CMD sh -c "alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
