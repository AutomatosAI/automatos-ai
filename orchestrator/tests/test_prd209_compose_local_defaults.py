"""PRD-209 S7 — the local edition is the compose default.

A human's plain ``docker compose up`` must boot the local edition, not abort
demanding Clerk keys. Today compose sets no ``AUTH_EDITION``/``env_file``, and the
committed ``envs/api.defaults`` still speaks the never-landed PRD-150 vocabulary
(``EDITION=oss`` / ``AUTH_PROVIDER=local``) with no consumer. This guard proves the
wiring is in place and honest:

* backend service consumes ``envs/api.defaults``; frontend consumes ``envs/frontend.defaults``;
* ``api.defaults`` speaks the shipped flag (``AUTH_EDITION=local``) + a non-empty
  ``DEFAULT_WORKSPACE_ID`` equal to the CI seed convention, and carries NO dead vocab;
* the three real secrets stay ``:?``-required in ``environment:`` (never moved to a
  committed file); Clerk vars are non-blocking (no ``:?``) so their absence is fine
  in local mode; and no secret VALUES leak into the committed ``envs/*.defaults``.

Pure/static — parses the compose YAML and reads the env files; no Docker, no boot.
"""
from __future__ import annotations

import pathlib
import re

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_API_DEFAULTS = _REPO_ROOT / "envs" / "api.defaults"
_FRONTEND_DEFAULTS = _REPO_ROOT / "envs" / "frontend.defaults"

# Q6 — the fixed well-known local workspace id, the exact value the CI seed
# (orchestrator/scripts/init_test_db.py, driven by test.yml) creates. UUID, to
# match workspaces.id. Keep in lock-step with envs/api.defaults + the smoke script.
CANONICAL_WORKSPACE_ID = "00000000-0000-0000-0000-0000000000c1"

# The three secrets that MUST stay hard-required and MUST NOT move into a file.
REQUIRED_SECRETS = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY")


def _compose() -> dict:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


def _env_files(service: dict) -> list[str]:
    ef = service.get("env_file", [])
    if isinstance(ef, str):
        ef = [ef]
    out = []
    for e in ef:
        out.append(e["path"] if isinstance(e, dict) else e)
    return out


def test_backend_consumes_api_defaults():
    svc = _compose()["services"]["backend"]
    assert any(
        "envs/api.defaults" in f for f in _env_files(svc)
    ), "backend service must declare env_file: envs/api.defaults"


def test_frontend_consumes_frontend_defaults():
    svc = _compose()["services"]["frontend"]
    assert any(
        "envs/frontend.defaults" in f for f in _env_files(svc)
    ), "frontend service must declare env_file: envs/frontend.defaults"


def test_api_defaults_speaks_shipped_vocab():
    text = _API_DEFAULTS.read_text(encoding="utf-8")
    assert re.search(r"^AUTH_EDITION=local\s*$", text, re.M), "api.defaults must set AUTH_EDITION=local"
    m = re.search(r"^DEFAULT_WORKSPACE_ID=(\S+)\s*$", text, re.M)
    assert m and m.group(1).strip(), "api.defaults must set a non-empty DEFAULT_WORKSPACE_ID"
    assert m.group(1).strip() == CANONICAL_WORKSPACE_ID, (
        f"DEFAULT_WORKSPACE_ID must equal the CI seed convention {CANONICAL_WORKSPACE_ID}, "
        f"got {m.group(1)!r}"
    )
    # Dead PRD-150 vocabulary must be gone (no backward-compat shim).
    assert not re.search(r"^EDITION=", text, re.M), "dead vocab EDITION= must be removed"
    assert not re.search(r"^AUTH_PROVIDER=", text, re.M), "dead vocab AUTH_PROVIDER= must be removed"


def test_frontend_defaults_speaks_shipped_vocab():
    text = _FRONTEND_DEFAULTS.read_text(encoding="utf-8")
    assert re.search(r"^NEXT_PUBLIC_AUTH_EDITION=local\s*$", text, re.M), (
        "frontend.defaults must set NEXT_PUBLIC_AUTH_EDITION=local (read by lib/auth-edition.ts)"
    )
    assert not re.search(r"^NEXT_PUBLIC_EDITION=", text, re.M), "dead vocab NEXT_PUBLIC_EDITION= must be removed"


def test_three_secrets_stay_required_in_compose():
    raw = _COMPOSE.read_text(encoding="utf-8")
    for secret in REQUIRED_SECRETS:
        assert re.search(rf"\$\{{{secret}:\?", raw), (
            f"{secret} must stay `:?`-required in docker-compose.yml environment (never moved to a file)"
        )


def test_clerk_vars_do_not_block_local_boot():
    raw = _COMPOSE.read_text(encoding="utf-8")
    # No Clerk var may be `:?` (that would abort compose config when Clerk is absent).
    for m in re.finditer(r"\$\{(\w*CLERK\w*):\?", raw):
        raise AssertionError(f"Clerk var {m.group(1)} is `:?`-required — blocks local boot")


def test_no_secret_values_in_defaults_files():
    secret_key = re.compile(r"^\s*([A-Z0-9_]*(PASSWORD|SECRET|TOKEN|PRIVATE_KEY|ACCESS_KEY)[A-Z0-9_]*)\s*=\s*(\S.*)$")
    for path in (_API_DEFAULTS, _FRONTEND_DEFAULTS):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = secret_key.match(line)
            assert not m, f"{path.name}:{i} carries a secret value: {m.group(1)}=… (public repo — keep in root .env)"
