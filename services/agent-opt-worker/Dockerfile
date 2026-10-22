# agent-opt-worker (PRD-58): the FutureAGI SDKs (agent-opt, ai-evaluation) in their
# own container. Only the API calls it, with X-Internal-Token (PRD-262 US-006).
#
# Railway builds this file with no build args and no target, so the LAST stage is
# production. CI also builds `test`, which adds pytest on top of the runtime
# (the agent-opt-worker job in .github/workflows/test.yml).
#
# No BuildKit cache mount: Railway fails one that lacks a hard-coded
# id=s/<service-id>-<path>, and this service's id is not known yet (PRD-262 §10).
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS runtime

# The runtime user first, so the code below is copied in already owned by it
# (no `chown -R` layer that copies every file again).
RUN useradd -m -u 1000 worker \
    && install -d -o worker -g worker /app

WORKDIR /app

# The lock is the whole resolution, every pin with its hashes, so pip installs
# exactly it: --require-hashes, and --no-deps so nothing outside the lock is
# fetched (overrides.txt keeps the PyPI package `fi` out; requirements.in).
COPY requirements.lock .
# eMudhra/emSign CA compromise (Oct 2026): pip < 26.2.2 keeps trusting the
# broken intermediate CAs via its own vendored certifi regardless of what
# the lock resolves to — must upgrade pip itself first. Hash-pinned so the
# upgrade download itself isn't trusting the very CA chain it's fixing.
RUN printf 'pip==26.2.2 --hash=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70\n' > /tmp/pip-boot.txt && pip install --no-cache-dir --require-hashes -r /tmp/pip-boot.txt
RUN pip install --no-cache-dir --require-hashes --no-deps -r requirements.lock

COPY --chown=1000:1000 . .
USER worker

EXPOSE 8080

# main.py refuses to start without AGENT_OPT_WORKER_TOKEN (worker_config.py).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# The service's own suite runs inside the image.
FROM runtime AS test
USER root
RUN pip install --no-cache-dir -r requirements-test.txt
USER worker

FROM runtime AS production
