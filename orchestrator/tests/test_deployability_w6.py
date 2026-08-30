"""PRD-176 (Wave 6) — deployability & reliability baseline.

Pure, dependency-light checks for the boot substrate that a fresh clone relies
on. These do NOT require Docker: they assert the *shape* of the compose file and
the entrypoint script (the two things that silently broke the fresh-clone boot),
plus that the config S3 seam MinIO uses is present.

- F009: the compose initdb mount points at a schema file that exists on disk.
- F051: docker-entrypoint.sh runs wait -> migrate -> seed -> start, and a failed
        migration aborts startup (does not fall through to `exec "$@"`).
- F089: the S3_ENDPOINT_URL seam MinIO is wired through exists in config, and the
        compose file defines a minio service + wires the backend to it.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

_ORCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ORCH_ROOT.parent
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_ENTRYPOINT = _REPO_ROOT / "docker-entrypoint.sh"


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Contain the config-reload blast radius (mirrors test_config_env_centralization).

    The F089 tests below ``importlib.reload(config)`` with S3_ENDPOINT_URL set /
    unset. Without a restore, the mutated config singleton + swept env bleed into
    downstream suites co-run after this file (test_harness_commands,
    test_prd143_concierge_journey). Snapshot os.environ and the config module
    reference at setup; restore both at teardown so nothing leaks past this file.
    """
    env_snapshot = dict(os.environ)
    saved = sys.modules.get("config")
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def _load_compose() -> dict:
    with _COMPOSE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# F009 — the initdb mount source path exists on disk
# ---------------------------------------------------------------------------


def test_compose_file_exists():
    assert _COMPOSE.is_file(), f"docker-compose.yml missing at {_COMPOSE}"


def test_compose_is_valid_yaml():
    compose = _load_compose()
    assert "services" in compose
    assert "postgres" in compose["services"]


def _initdb_mount_source() -> str:
    compose = _load_compose()
    volumes = compose["services"]["postgres"].get("volumes", [])
    for vol in volumes:
        # bind mounts are "src:dst[:mode]" strings
        if isinstance(vol, str) and "docker-entrypoint-initdb.d" in vol:
            return vol.split(":", 1)[0]
    return None


def test_no_initdb_schema_mount_remains():
    """PRD-209 S2 revision (2026-08-29): the initdb SQL snapshot is retired — fresh
    databases are built by scripts/init_fresh_db.py from the entrypoint instead.
    Postgres must start EMPTY; any initdb.d schema mount reintroduces the stale-copy
    writer this wave deleted (fresh clones were getting 107 of prod's ~152 tables)."""
    src = _initdb_mount_source()
    assert src is None, (
        f"postgres service mounts an initdb.d schema again ({src!r}) — fresh schema "
        "comes from init_fresh_db via the entrypoint, not an initdb snapshot"
    )


# ---------------------------------------------------------------------------
# F051 — the entrypoint migrates, in order, failing closed
# ---------------------------------------------------------------------------


def _entrypoint_text() -> str:
    return _ENTRYPOINT.read_text(encoding="utf-8")


def test_entrypoint_exists():
    assert _ENTRYPOINT.is_file(), f"docker-entrypoint.sh missing at {_ENTRYPOINT}"


def test_entrypoint_runs_alembic_upgrade():
    """F051: the entrypoint must actually migrate (it never did before W6)."""
    text = _entrypoint_text()
    assert "alembic upgrade" in text, "entrypoint does not run alembic upgrade"


def test_entrypoint_lifecycle_order_wait_migrate_seed_start():
    """F051: order is wait -> migrate -> seed -> start (exec)."""
    text = _entrypoint_text()
    # Anchor on the main-execution call sites, not the function definitions.
    wait_at = text.rindex("wait_for_postgres")
    migrate_at = text.rindex("run_migrations")
    seed_at = text.rindex("load_seed_data")
    exec_at = text.rindex('exec "$@"')
    assert wait_at < migrate_at < seed_at < exec_at, (
        "entrypoint lifecycle must be wait -> migrate -> seed -> start; "
        f"offsets wait={wait_at} migrate={migrate_at} seed={seed_at} exec={exec_at}"
    )


def test_entrypoint_migration_fails_closed():
    """F051: a failed migration must abort startup, not continue on a half schema.

    The seed step is deliberately lenient ("continue anyway"); the migrate step
    must NOT be — it exits non-zero on failure so the app never starts against a
    half-built schema.
    """
    text = _entrypoint_text()
    # Isolate the run_migrations function body.
    m = re.search(r"run_migrations\(\)\s*\{(.*?)\n\}", text, re.DOTALL)
    assert m, "run_migrations() function not found"
    body = m.group(1)
    assert "exit 1" in body, "run_migrations must exit non-zero on failure (fail-closed)"
    # And it must not swallow the failure with the seed step's leniency phrasing.
    assert "continue anyway" not in body.lower(), (
        "migrate step must fail closed, not 'continue anyway'"
    )


# ---------------------------------------------------------------------------
# F089 — the S3_ENDPOINT_URL seam MinIO rides on, + the minio service
# ---------------------------------------------------------------------------


def test_config_exposes_s3_endpoint_url():
    """F089: MinIO is wired via the S3_ENDPOINT_URL boto endpoint override."""
    import config as cfg

    assert hasattr(cfg.config, "S3_ENDPOINT_URL"), (
        "config must expose S3_ENDPOINT_URL for the local object-store seam"
    )


def test_s3_endpoint_url_defaults_empty_and_reads_env(monkeypatch):
    """Default unset (prod uses real S3); env override selects local MinIO."""
    import importlib
    import sys

    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    sys.modules.pop("config", None)
    import config as cfg

    importlib.reload(cfg)
    assert cfg.config.S3_ENDPOINT_URL == ""

    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    sys.modules.pop("config", None)
    import config as cfg2

    importlib.reload(cfg2)
    assert cfg2.config.S3_ENDPOINT_URL == "http://minio:9000"


def test_compose_defines_minio_service():
    compose = _load_compose()
    assert "minio" in compose["services"], "compose must define a local minio service"
    minio = compose["services"]["minio"]
    assert "minio" in minio.get("image", ""), "minio service should use a minio image"


def test_compose_backend_wired_to_minio_endpoint():
    """Backend env carries S3_ENDPOINT_URL so the flywheel upload targets MinIO."""
    compose = _load_compose()
    backend_env = compose["services"]["backend"].get("environment", {})
    # environment can be a dict or a list of "K=V" strings; normalize to keys.
    if isinstance(backend_env, list):
        keys = {item.split("=", 1)[0] for item in backend_env}
    else:
        keys = set(backend_env.keys())
    assert "S3_ENDPOINT_URL" in keys, "backend must receive S3_ENDPOINT_URL for MinIO"


def test_compose_defines_minio_bucket_init():
    """A one-shot service creates the documents bucket the flywheel writes to."""
    compose = _load_compose()
    assert "minio-init" in compose["services"], (
        "compose must create the flywheel's bucket on first boot (minio-init)"
    )


# ---------------------------------------------------------------------------
# F089 — DocumentManager threads S3_ENDPOINT_URL into its boto client
# ---------------------------------------------------------------------------
# The flywheel's upload path (get_document_manager -> DocumentManager) is what
# fail-softs to None with no object store. This proves the endpoint override the
# whole MinIO wiring depends on actually reaches boto3.client — without a live
# server (we spy on boto3.client and stub the heavy post-client collaborators).
# Since PRD-233 S4 the client is built by core.storage on FIRST USE, so the
# probe reaches through the factory: env -> config -> factory -> boto3.client.


def _make_document_manager_capturing_boto(monkeypatch, endpoint_url: str):
    import importlib
    import sys

    import boto3

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv("S3_ENDPOINT_URL", endpoint_url) if endpoint_url else monkeypatch.delenv(
        "S3_ENDPOINT_URL", raising=False
    )
    monkeypatch.delenv("S3_USE_PATH_STYLE", raising=False)
    # The prod shape (no endpoint) is "configured" through an explicit key pair.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    sys.modules.pop("config", None)
    import config  # noqa: F401

    importlib.reload(config)

    import core.storage.s3 as s3mod
    from modules.rag.ingestion import manager as mgr_mod

    # The factory bound its config instance at import; point it at the one just
    # reloaded from this test's env and drop any memoized client.
    monkeypatch.setattr(s3mod, "config", config.config)
    s3mod.reset_s3_client()

    captured = {}

    def _spy_client(service, **kwargs):
        captured["service"] = service
        captured["kwargs"] = kwargs
        return MagicMock()

    # Spy on the boto client; neutralize the heavy collaborators __init__ calls.
    monkeypatch.setattr(boto3, "client", _spy_client)
    monkeypatch.setattr(
        "core.llm.create_embedding_manager",
        lambda *a, **kw: _StubEmbeddings(),
    )

    manager = mgr_mod.DocumentManager(db_config={}, workspace_id="ws-test")
    assert not captured, "DocumentManager must not build an S3 client at construction"
    manager.s3_client  # first use builds it through the factory
    s3mod.reset_s3_client()
    return captured


class _StubEmbeddings:
    def get_provider_info(self):
        return {"provider": "stub"}


def test_document_manager_uses_minio_endpoint_when_set(monkeypatch):
    """With S3_ENDPOINT_URL set, boto3.client receives it (upload -> MinIO)."""
    captured = _make_document_manager_capturing_boto(monkeypatch, "http://minio:9000")
    assert captured["service"] == "s3"
    assert captured["kwargs"].get("endpoint_url") == "http://minio:9000"


def test_document_manager_no_endpoint_when_unset(monkeypatch):
    """With S3_ENDPOINT_URL unset, endpoint_url is None (prod real AWS S3)."""
    captured = _make_document_manager_capturing_boto(monkeypatch, "")
    assert captured["service"] == "s3"
    assert captured["kwargs"].get("endpoint_url") is None
