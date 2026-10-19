# syntax=docker/dockerfile:1
# =============================================================================
# Automatos AI - Backend Dockerfile (Multi-stage)
# =============================================================================
# Stages:
#   - uv: the installer pybuild copies (pinned, read by Dependabot)
#   - pybuild: installs the locked dependencies; never ships
#   - base: Common dependencies
#   - development: Hot reload; no code (docker-compose.dev.yml, `make dev`, mounts
#     the source over /app)
#   - compile: the code, with its boot modules compiled; never ships itself
#   - production: Optimized, minimal image
# =============================================================================

# -----------------------------------------------------------------------------
# uv, the installer pybuild runs (PRD-262 US-022). A stage of its own, so the
# tag@digest pin sits on a FROM line, where Dependabot reads it (FR-9). Only the
# binary is copied, into pybuild, which never ships.
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv

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

COPY --from=uv /uv /usr/local/bin/uv

# One lock for every build, Railway's included (PRD-262 US-027): the Knowledge Graph's
# Leiden clustering comes from graspologic-native (modules/knowledge/leiden_partition.py),
# which requirements.lock pins, so no build arg picks a second lock with graphifyy's
# [leiden] extra and its ~585 MB any more.
COPY requirements.lock scripts/split_lock.py ./

# The lock, exactly (PRD-262 US-022): every pin with its hashes, so CI, GHCR and
# Railway install the same versions, and --no-deps resolves nothing. The lock goes in
# two halves, each into its own prefix, which the base stage copies as its own layer:
# first the stable set split_lock.py names (numpy, scipy, grpcio, cryptography,
# SQLAlchemy, Pillow, lxml, fonttools, uvloop, asyncpg, pydantic-core), then the
# rest, so a bump outside the stable set leaves the first layer as it was.
# --prefix keeps each half in a tree of its own; the scripts it writes run
# /usr/local/bin/python, where the base stage puts both trees.
# split_lock.py leaves two sets out of both halves (PRD-262, the 19 Oct breakdown's rows
# #11 and #47). LEFT_OUT, which no image installs: the 12 tree-sitter grammars graphifyy
# requires that only graphify/extract.py imports, a module the app never loads (about
# 47 MB; the lock still pins them, because CI's pip --require-hashes refuses a
# requirement it does not pin). DEVELOPMENT_ONLY, watchfiles, which only uvicorn --reload
# uses: a third prefix, /install/development, that only the development stage copies.
# No bytecode here: the compile below writes it once the prune is done.
# Wheels only (PRD-262 US-015): a package that would build from source fails the
# build instead. The one exception is a pure-Python sdist: pysher (composio's);
# graspologic left the list with the [leiden] extra (PRD-262 US-027). pip's
# `--only-binary :all:` cleared every per-package exception named before it, so
# `--no-binary` still comes after; uv lets a named --no-binary package build either
# way.
# No BuildKit cache mount: Railway fails one that lacks a hard-coded
# id=s/<service-id>-<path>, and the service ids are not known (PRD-262 §10).
RUN python split_lock.py requirements.lock stable.txt rest.txt development.txt && \
    for part in stable rest development; do \
        uv pip install --python /usr/local/bin/python --prefix "/install/$part" --no-deps --require-hashes \
            --no-compile-bytecode --no-cache --only-binary :all: --no-binary pysher \
            -r "$part.txt" || exit 1; \
    done

# What runs, and nothing else (PRD-262 US-020). strip --strip-unneeded drops the .debug_*
# sections from every compiled library (about 91 MB of the 16 Oct pip layer) and, since
# the 19 Oct breakdown's row #25, the static symbol table too (.symtab and .strtab, about
# 18 MB more: scipy, cryptography, pypdfium2_raw, tiktoken …). It leaves .dynsym, which
# the loader, dlsym and ctypes resolve symbols through, as it is; a native crash's
# backtrace loses the names of local functions, nothing else. It skips the auditwheel
# *.libs directories: they hold about 1 MB of it, in libraries auditwheel renamed and
# rewrote with patchelf, the kind of file strip has broken before. The extensions that
# link them were rewritten too, so CI loads every file the strip touched
# (scripts/ci/image_site_packages.py). Then every directory named exactly `tests` under
# site-packages goes (about 108 MB); numpy.testing, sqlalchemy.testing and
# alembic.testing are importable APIs, not test suites, and stay. /install holds both
# halves of the lock (/install/stable and /install/rest), so this covers both.
RUN find /install -type f -name '*.so*' -not -path '*.libs/*' -exec strip --strip-unneeded {} + && \
    find /install -type d -name tests -path '*/site-packages/*' -prune -exec rm -rf {} +

# The C, C++ and Cython sources and headers in site-packages go (PRD-262, the 19 Oct
# breakdown's row #37, about 11.6 MB): fontTools' generated .c, scipy's .pyx and .pxd,
# lxml's .h and .pxi, numpy's headers. Only a compiler reads them (numpy.get_include(),
# Cython's cimport), the runtime has none, and no package opens one when it imports or
# runs; scripts/ci/image_site_packages.py fails on one left behind.
RUN find /install -type f -path '*/site-packages/*' \( -name '*.c' -o -name '*.cpp' -o -name '*.h' \
        -o -name '*.hpp' -o -name '*.pyx' -o -name '*.pxd' -o -name '*.pxi' \) -delete

# scipy keeps the subpackages the code reaches, and no others (PRD-262, the 19 Oct
# breakdown's row #17, about 29 MB). Nothing in the app imports scipy but
# core/math/probability_theory.py (Student's t, from scipy.special); it is in the lock
# for datasketch, which graphify's dedup runs (MinHashLSH integrates with
# scipy.integrate.quad; WeightedMinHash takes scipy.sparse), and graspologic-native,
# which never imports it, and networkx's pagerank reaches scipy.sparse. Those import
# _lib, constants, integrate, linalg, optimize, sparse, spatial and special when they
# load, and none of these imports a deleted one when it loads: only solve_bvp, qmc_quad,
# differential_evolution, shgo and RotationSpline do, inside the function, and nothing
# calls them. A kept subpackage missing fails the build. scipy is in the first half of
# the lock; scripts/ci/image_scipy.py runs every scipy path of the code in the image.
RUN scipy=/install/stable/lib/python3.11/site-packages/scipy && \
    for kept in _lib constants integrate linalg optimize sparse spatial special; do \
        test -d "$scipy/$kept" || { echo "scipy has no $kept" >&2; exit 1; }; \
    done && \
    for dropped in cluster datasets differentiate fftpack interpolate io misc ndimage odr signal stats; do \
        rm -rf "${scipy:?}/$dropped" || exit 1; \
    done

# botocore keeps the models of the services we call, and no others (PRD-262 US-021).
# botocore/data holds one for every AWS service, 437 directories and about 18.6 MB.
# The app builds s3, s3vectors and bedrock-runtime clients; botocore's credential
# providers build sts (assumed roles, web identity), sso and sso-oidc (an SSO profile).
# The top-level files (endpoints, partitions, retry and default configuration) stay.
# A kept service missing from the install fails the build before anything goes. Any
# other client raises UnknownServiceError: test_prd262_botocore_services.py fails when
# code in orchestrator/ names one, and CI builds each kept client inside the image
# (scripts/ci/image_botocore.py). botocore is in the second half of the lock.
RUN keep="s3 s3vectors bedrock-runtime sts sso sso-oidc" && \
    data=/install/rest/lib/python3.11/site-packages/botocore/data && \
    for service in $keep; do \
        test -d "$data/$service" || { echo "botocore/data has no $service" >&2; exit 1; }; \
    done && \
    for dir in "$data"/*/; do \
        name=$(basename "$dir"); \
        case " $keep " in *" $name "*) ;; *) rm -rf "$dir" || exit 1 ;; esac; \
    done

# site-packages' bytecode, once the prune is done (PRD-262 US-022), for each half and the
# development part. -s and -p record each module under /usr/local, where the base and
# development stages put it.
# unchecked-hash: Python never compares a .pyc with its source's mtime, so the clamped
# mtimes of images.yml's reproducible build (SOURCE_DATE_EPOCH, rewrite-timestamp) can't
# make one stale, and the automatos user, who can't write to /usr/local, never
# recompiles one at start-up (scripts/ci/image_site_packages.py counts `bytecode is
# stale` under `python -v -c 'import main'`). uv has no invalidation-mode option, so
# compileall writes it. Nothing is excluded with -x: pip compiled every .py of the
# published edge's site-packages (951a84b) on its Python 3.11, and the .py of every
# package the lock moved parse as Python 3.11 (checked 19 Oct; graspologic-native, which
# US-027 added, has one .py, an __init__ that re-exports its compiled module). A file
# that ever fails to compile fails the build; exclude it here by name, with the reason,
# only if it is a template or test data. PYTHONHASHSEED=0 fixes the string
# hashes, the one input of a .pyc that changes from run to run, so the same sources
# compile to the same bytes and the layer stays reproducible.
RUN for prefix in /install/stable /install/rest /install/development; do \
        PYTHONHASHSEED=0 python -m compileall -q -s "$prefix" -p /usr/local --invalidation-mode unchecked-hash \
            "$prefix/lib/python3.11/site-packages" || exit 1; \
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

# The two halves of the lock, each its own layer, the stable set first (PRD-262
# US-022): a bump outside it downloads only the second.
COPY --from=pybuild /install/stable /usr/local
COPY --from=pybuild /install/rest /usr/local

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

# No application code (PRD-262 US-017). docker-compose.dev.yml (`make dev`) bind-mounts
# ./orchestrator over /app, which hid a baked copy, while `COPY . .` and a `chown -R /app`
# stored the whole build context twice on every build, a checkout's uploads and caches
# included. Run this stage with the source mounted at /app. The default compose stack
# runs the production stage below (PRD-262 US-035).
#
# The user and the directories the app writes, owned by it (no -R: they are empty), so
# a fresh backend_logs or backend_data named volume mounted on one starts writable
# (PRD-209: the credential key lives in /app/data). /workspaces too (PRD-262 US-040): a
# fresh automatos_workspaces_data volume takes the owner of the directory it is first
# mounted on, so it is uid 1000 whether the backend or the workspace worker mounts it
# first. The core stack (docker-compose.core.yml) starts the backend without the worker.
# The production stage's first RUN, word for word, so the two stages share the layer.
RUN useradd -m -u 1000 automatos && \
    mkdir -p /app/logs /app/vector_stores /app/projects /app/exports /app/data /workspaces && \
    chown automatos:automatos /app /app/logs /app/vector_stores /app/projects /app/exports /app/data /workspaces

# watchfiles, through which --reload below watches the mounted source (PRD-262 row #47):
# split_lock.py's DEVELOPMENT_ONLY, from the lock, which the production stage never
# copies. Without it uvicorn would poll every .py under /app four times a second.
COPY --from=pybuild /install/development /usr/local

# The wait-migrate-seed entrypoint, the only owner of migrations (PRD-262 US-029):
# before the server it runs the migration step, plus the seeds where
# AUTOMATOS_MIGRATE_ON_BOOT=true (compose sets it) asks. Compose also bind-mounts
# it, so edits apply without a rebuild.
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# The healthcheck (F416): the HEALTHCHECK below and the entrypoint's watchdog both run it.
COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh
# The placeholder-secret guard the entrypoint runs before a command (PRD-262 US-037).
COPY --chmod=0755 docker-secrets-guard.sh /usr/local/bin/docker-secrets-guard.sh
USER automatos

# Expose port
EXPOSE 8000

# Health check. While the server starts, Docker probes every second instead of every
# 30 s (--start-interval, Docker Engine 25 or later; the syntax directive on line 1 lets
# older builders read it), so the container reports healthy within about a second of
# /health answering (PRD-262 US-038). Compose sets its own (docker-compose.yml);
# Railway runs its own health check, not this one.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --start-interval=1s --retries=3 \
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
# useradd gives the automatos group the user's id, 1000. /workspaces is the mount point
# of the workspaces volume: owned by 1000 here, a fresh volume is the user's whichever
# service mounts it first (PRD-262 US-040, as in the development stage).
RUN useradd -m -u 1000 automatos && \
    mkdir -p /app/logs /app/vector_stores /app/projects /app/exports /app/data /workspaces && \
    chown automatos:automatos /app /app/logs /app/vector_stores /app/projects /app/exports /app/data /workspaces

# The stdlib's bytecode, compiled once as root (PRD-262 US-014): the python base
# ships none, and the automatos user can't write to /usr/local, so without it
# every process compiles each stdlib module it imports, every time. site-packages
# is excluded: its bytecode comes only from pybuild. So are idlelib, tkinter,
# turtledemo, lib2to3 and pydoc_data (PRD-262, the 19 Oct breakdown's row #47, about
# 2.65 MB): tkinter can't import here (python:slim has no Tk libraries), nothing runs
# IDLE, turtledemo or 2to3, and pydoc's help() compiles pydoc_data in memory if a
# person ever asks. unchecked-hash .pyc stay valid whatever the build clamps file
# mtimes to. Before the COPYs, so neither a code change nor an entrypoint edit
# rebuilds this layer.
RUN python -m compileall -q -x '/site-packages/|/(idlelib|tkinter|turtledemo|lib2to3|pydoc_data)/' --invalidation-mode unchecked-hash /usr/local/lib/python3.11

# The wait-migrate-seed entrypoint, the only owner of migrations (PRD-262 US-029).
# Before the server it runs the migration step, so Railway (no flag set) migrates
# there; AUTOMATOS_MIGRATE_ON_BOOT=true adds the seeds (compose), and
# AUTOMATOS_SKIP_MIGRATIONS=true runs nothing (the Helm api pods).
# `docker-entrypoint.sh migrate` runs the full lifecycle and exits (the Kubernetes
# migration Job).
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# The healthcheck (F416): the HEALTHCHECK below and the entrypoint's watchdog both run it.
COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh
# The placeholder-secret guard the entrypoint runs before a command (PRD-262 US-037).
COPY --chmod=0755 docker-secrets-guard.sh /usr/local/bin/docker-secrets-guard.sh

# The application code, with its boot modules' bytecode from the compile stage
# (PRD-262 US-032): the stage's last filesystem layer, code and bytecode together,
# so a code-only update downloads this one layer (PRD-262 US-016; images.yml
# judges it against .github/image-budgets.json's update bar). Owned by the user
# created above, by its numeric id. Everything after it only sets image metadata.
# The test tools are in requirements-dev.txt (locked in requirements-dev.lock),
# which only CI installs; .dockerignore keeps the rest of the context out.
COPY --from=compile --chown=1000:1000 /app /app
USER automatos

# Expose port
EXPOSE 8000

# Health check. While the server starts, Docker probes every second instead of every
# 30 s (--start-interval, Docker Engine 25 or later; the syntax directive on line 1 lets
# older builders read it), so the container reports healthy within about a second of
# /health answering (PRD-262 US-038). Compose sets its own (docker-compose.yml);
# Railway runs its own health check, not this one.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --start-interval=1s --retries=3 \
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
