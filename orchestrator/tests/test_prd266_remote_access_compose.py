"""PRD-266 US-005a (§4.3, §10) — the compose file publishes only the API and the web app.

* ``API_PUBLISH_ADDRESS`` and ``WEB_PUBLISH_ADDRESS`` publish the backend and the frontend,
  each defaulting to ``${BIND_ADDRESS:-127.0.0.1}``, so a fresh clone stays on loopback.
* No Postgres, Redis, MinIO or Qdrant port (nor Adminer's or SearXNG's) can follow a publish
  variable: pairing a phone never opens the database. Checked on every tracked compose file.
* The two addresses and ``LOCAL_REMOTE_ACCESS`` (default ``off``) reach the backend's
  environment, so its boot check (``services.mobile.remote_access``) knows whether it is
  published and who may call it; a ``.env`` value reaches a container only that way.
* The header documents the three beside ``BIND_ADDRESS``, and the file stays under 800 lines.

Pure: the compose files are read as YAML, nothing is started.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "docker-compose.yml"
BIND = "${BIND_ADDRESS:-127.0.0.1}"
API_PUBLISH = "${API_PUBLISH_ADDRESS:-" + BIND + "}"
WEB_PUBLISH = "${WEB_PUBLISH_ADDRESS:-" + BIND + "}"
PUBLISH_VARIABLES = ("API_PUBLISH_ADDRESS", "WEB_PUBLISH_ADDRESS")
DATASTORES = ("postgres", "redis", "minio", "qdrant", "adminer", "searxng")
MAX_LINES = 800


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that reads compose's ``!reset`` / ``!override`` merge tags as plain values."""


def _untagged(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


for _tag in ("!reset", "!override"):
    _ComposeLoader.add_constructor(_tag, _untagged)


def _services(path: Path) -> dict:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader) or {}
    return (data.get("services") if isinstance(data, dict) else None) or {}


def _compose_files() -> list:
    proc = subprocess.run(["git", "ls-files"], cwd=str(REPO), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    names = [n for n in proc.stdout.splitlines() if Path(n).name.startswith(("docker-compose", "compose"))
             and n.endswith((".yml", ".yaml"))]
    return [REPO / n for n in names]


def _host_address(entry: object) -> str:
    """The host-address part of a ``ports:`` entry (short or long syntax), as written."""
    if isinstance(entry, dict):
        return str(entry.get("host_ip") or "")
    text = str(entry)
    for wrapped in (API_PUBLISH, WEB_PUBLISH, BIND):
        if text.startswith(wrapped + ":"):
            return wrapped
    return text.rsplit(":", 2)[0] if text.count(":") >= 2 else ""


def _ports(service: str) -> list:
    return list((_services(COMPOSE).get(service) or {}).get("ports") or [])


def test_the_api_and_the_web_app_publish_on_their_own_variable_defaulting_to_bind():
    assert [_host_address(e) for e in _ports("backend")] == [API_PUBLISH]
    assert [_host_address(e) for e in _ports("frontend")] == [WEB_PUBLISH]
    # The fresh-clone smoke still finds the web app on loopback by default.
    assert WEB_PUBLISH + ":${FRONTEND_PORT:-3000}:3000" in _ports("frontend")


@pytest.mark.parametrize("service", DATASTORES)
def test_every_datastore_port_stays_on_bind_exactly(service):
    ports = _ports(service)
    assert ports, f"{service} publishes no port: the check would pass for nothing"
    assert {_host_address(e) for e in ports} == {BIND}


def test_no_datastore_port_can_follow_a_publish_variable_in_any_compose_file():
    """§10's CI test: in every tracked compose file, only the backend and the frontend
    name a publish variable in their ports."""
    offenders = []
    for path in _compose_files():
        for name, service in _services(path).items():
            if name in ("backend", "frontend"):
                continue
            for entry in (service or {}).get("ports") or []:
                if any(variable in str(entry) for variable in PUBLISH_VARIABLES):
                    offenders.append(f"{path.relative_to(REPO)} · {name}: {entry!r}")
    assert COMPOSE in _compose_files()
    assert not offenders, "a datastore follows a publish variable:\n" + "\n".join(offenders)


def test_the_three_variables_reach_the_backends_environment():
    env = (_services(COMPOSE)["backend"] or {}).get("environment") or {}
    assert env.get("API_PUBLISH_ADDRESS") == API_PUBLISH
    assert env.get("WEB_PUBLISH_ADDRESS") == WEB_PUBLISH
    assert env.get("LOCAL_REMOTE_ACCESS") == "${LOCAL_REMOTE_ACCESS:-off}"


def test_the_frontend_gets_no_remote_access_switch():
    """The backend decides who may call it; the web app is only told by discovery (US-001)."""
    env = (_services(COMPOSE)["frontend"] or {}).get("environment") or {}
    assert "LOCAL_REMOTE_ACCESS" not in env


def test_the_header_documents_the_three_beside_bind_address_and_the_file_stays_small():
    text = COMPOSE.read_text(encoding="utf-8")
    header = text.split("\nservices:", 1)[0]
    for name in ("BIND_ADDRESS", *PUBLISH_VARIABLES, "LOCAL_REMOTE_ACCESS", "mint_pairing_code"):
        assert name in header, f"the header does not document {name}"
    assert len(text.splitlines()) < MAX_LINES
