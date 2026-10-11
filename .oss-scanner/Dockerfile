# Build image for Anthropic's OSS Scanner (https://github.com/anthropics/oss-scanner).
# The scanner builds this with the repository root as the context, then audits the
# image with no network. Everything the audit needs (Python and Node dependencies,
# a local PostgreSQL for the backend tests) is fetched here.
# Mirrors the CI lanes in .github/workflows/test.yml: orchestrator-tests,
# frontend-ci and cli-host-tests.
FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NEXT_TELEMETRY_DISABLED=1

# System libraries the orchestrator imports (WeasyPrint, python-magic, OCR, PDF) and a
# local PostgreSQL 15 (bookworm's) so the backend tests run offline.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libcairo2 libffi-dev \
      libmagic1 ghostscript tesseract-ocr fonts-dejavu-core \
      postgresql-15 postgresql-client-15 \
    && rm -rf /var/lib/apt/lists/*

# Node 20, the version the frontend CI lane runs.
COPY --from=node:20-bookworm /usr/local/bin/node /usr/local/bin/node
COPY --from=node:20-bookworm /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Scanner contract: the checkout lives at /src.
COPY . /src
WORKDIR /src

# Backend (orchestrator) and the Python services, as CI installs them.
RUN pip install -r orchestrator/requirements.txt \
    && pip install -r services/workspace-worker/requirements.txt \
    && pip install pytest-timeout

# Frontend: `npm install`, not `npm ci` (CI does the same; see frontend-ci in test.yml).
RUN cd frontend && npm install --no-audit --no-fund

# Local-only settings for offline reproduction. No production credentials exist here.
ENV DATABASE_URL=postgresql://test:test@127.0.0.1:5432/test_db \
    POSTGRES_USER=test \
    POSTGRES_PASSWORD=test \
    POSTGRES_HOST=127.0.0.1 \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=test_db \
    ENVIRONMENT=development \
    AUTH_EDITION=local \
    DEFAULT_WORKSPACE_ID=00000000-0000-0000-0000-0000000000c1

# Create the test database and its schema once, so `.oss-scanner/start-db.sh` only
# has to start the server.
RUN chmod +x .oss-scanner/start-db.sh \
    && .oss-scanner/start-db.sh \
    && (cd orchestrator && python scripts/init_test_db.py) \
    && service postgresql stop

# Cheap test runs so the build log shows the image works. A failing test does not stop
# the build; read the log rather than assuming a pass.
RUN .oss-scanner/start-db.sh \
    && (cd orchestrator && pytest tests -q -p no:cacheprovider --timeout=60 --timeout-method=thread \
          -k "security or ssrf or tenant or workspace_scope or url_validator or web_access or auth" \
        || echo "WARNING: orchestrator security-related tests reported failures") \
    ; service postgresql stop
RUN (cd services/cli-host && pytest -q tests) || echo "WARNING: cli-host tests reported failures"
RUN (cd frontend && npm run test) || echo "WARNING: frontend vitest reported failures"
