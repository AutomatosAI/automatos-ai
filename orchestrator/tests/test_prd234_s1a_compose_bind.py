"""PRD-234 S1a — source guards for the session-mode security posture.

* ``docker-compose.yml``: the backend-side ports (API, Postgres, Redis, MinIO)
  publish on ``${BIND_ADDRESS:-127.0.0.1}`` — this machine only, unless the
  operator sets ``BIND_ADDRESS`` deliberately. In session mode the API can run
  commands on the machine; a LAN-wide default would be a shell for the LAN.
* the pairing token can never be an empty default (the ``WORKER_INTERNAL_TOKEN=""``
  pattern is not copied): ``resolve_host_by_token`` refuses empty/None.
* the boot gate lives OUTSIDE ``run_stage`` (the only place an abort is real).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

REPO = _ORCH.parent
COMPOSE = REPO / "docker-compose.yml"

_LOOPBACK_PUBLISHED = {
    "POSTGRES_PORT": "5432",
    "REDIS_PORT": "6379",
    "MINIO_PORT": "9000",
    "MINIO_CONSOLE_PORT": "9001",
    "API_PORT": "8000",
}


def test_backend_ports_publish_on_loopback_by_default():
    text = COMPOSE.read_text(encoding="utf-8")
    for var, container_port in _LOOPBACK_PUBLISHED.items():
        pattern = re.compile(
            r'-\s+"\$\{BIND_ADDRESS:-127\.0\.0\.1\}:\$\{' + var + r':-\d+\}:' + container_port + r'"'
        )
        assert pattern.search(text), f"{var} must publish on ${{BIND_ADDRESS:-127.0.0.1}}"
        bare = re.compile(r'-\s+"\$\{' + var + r':-\d+\}:' + container_port + r'"')
        assert not bare.search(text), f"{var} still has an all-interfaces publish line"


def test_compose_documents_the_bind_dial():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "BIND_ADDRESS=0.0.0.0" in text, "the deliberate way to expose must be documented in the file"


def test_host_token_resolution_refuses_empty_tokens():
    from services import cli_host_service as svc

    class _NeverQueried:
        def query(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("an empty token must short-circuit before any lookup")

    assert svc.resolve_host_by_token(_NeverQueried(), "") is None
    assert svc.resolve_host_by_token(_NeverQueried(), None) is None


def test_boot_gate_runs_outside_run_stage():
    """The CLI gate is inside validate_auth_edition, which main.py calls directly —
    never wrapped by run_stage (the wrapper that swallowed the PRD-175 guard)."""
    cfg = (_ORCH / "config.py").read_text(encoding="utf-8")
    start = cfg.index("def validate_auth_edition(self)")
    body = cfg[start:cfg.index("def validate(self)", start)]
    assert "CLI_RUNTIME_ENABLED" in body and "AUTH_EDITION=local" in body
    main = (_ORCH / "main.py").read_text(encoding="utf-8")
    assert "config.validate_security()" in main
    assert "run_stage(report, BootstrapStage.DATABASE_INIT" in main


def test_claude_driver_invariants_are_not_yet_contradicted_in_backend_code():
    """S1b owns the host; the backend must not pre-empt the §Terms invariant by
    handling credentials itself. No backend module reads Claude Code credentials
    or sets the API key into a session environment."""
    forbidden = ("CLAUDE_CODE_OAUTH_TOKEN", ".credentials.json", "CLAUDE_CODE_ENTRYPOINT")
    for path in (
        _ORCH / "services" / "cli_host_service.py",
        _ORCH / "api" / "cli_hosts.py",
        _ORCH / "core" / "cli_runtime.py",
    ):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} must not reference {token}"


def test_compose_passes_the_session_mode_dial_into_the_backend():
    """A .env value only reaches a container through `environment:` (or the
    api.local lane). CLI_RUNTIME_ENABLED must be wired, default off."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "CLI_RUNTIME_ENABLED: ${CLI_RUNTIME_ENABLED:-false}" in text
