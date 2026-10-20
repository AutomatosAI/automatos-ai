"""PRD-262 US-035: compose runs the production backend; hot reload moves to `make dev`.

The default stack (docker-compose.yml) builds the backend's production stage, the one
CI builds, scans and publishes, and lays nothing of the checkout over it: no source,
entrypoint or healthcheck bind mount. docker-compose.dev.yml (`make dev`) keeps the
development stage, the source mount and uvicorn --reload, and takes over the entrypoint
and healthcheck mounts.

TESTER's night and test tooling keeps what it needs on the production-stage backend
(PRD-262 §9-Q1, FR-13): backend_data at /app/data, the backend_logs named volume at
/app/logs, both /workspaces mounts, and its env_file, envs/api.defaults plus the
optional envs/api.local. The worker count comes from WEB_CONCURRENCY=1 in
envs/api.defaults (US-029), so compose sets no command of its own.

Static reads of the compose files, the Makefile, the Dockerfile, envs/api.defaults and
the docs that say how to run the stack. The running stack is checked by the smoke
(scripts/ci/smoke_production_backend.py, test_prd262_smoke_production_backend.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO / "orchestrator" / "Dockerfile"
_API_DEFAULTS = _REPO / "envs" / "api.defaults"
_LOCAL_ROOT = "/workspaces/${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"
_SOURCE = "./orchestrator"
# The entrypoint, the healthcheck and the placeholder-secret guard (PRD-262 US-037).
_SCRIPTS = ("/usr/local/bin/docker-entrypoint.sh", "/usr/local/bin/docker-healthcheck.sh",
            "/usr/local/bin/docker-secrets-guard.sh")
_DEV_COMMAND = "docker compose -f docker-compose.yml -f docker-compose.dev.yml"


def _compose(name: str) -> dict:
    return yaml.safe_load((_REPO / name).read_text(encoding="utf-8"))


def _fields(entry: str) -> list:
    """A short-syntax volume split on its colons, never on one inside ``${VAR:-default}``."""
    fields, current, depth = [], "", 0
    for char in entry:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == ":" and depth == 0:
            fields, current = fields + [current], ""
        else:
            current += char
    return fields + [current]


def _mounts(service: dict) -> dict:
    """``{target: (source, mode)}`` of a service's short-syntax volumes."""
    out = {}
    for entry in service.get("volumes") or []:
        source, target, *mode = _fields(entry)
        out[target] = (source, mode[0] if mode else "rw")
    return out


@pytest.fixture(scope="module")
def base() -> dict:
    return _compose("docker-compose.yml")


@pytest.fixture(scope="module")
def dev() -> dict:
    return _compose("docker-compose.dev.yml")


def _stage_cmd(stage: str) -> str:
    """The CMD line of one Dockerfile stage, continuation lines joined."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    start = re.search(rf"^FROM \S+ as {stage}$", text, re.M | re.I)
    assert start, f"the Dockerfile has no {stage} stage"
    following = re.search(r"^FROM ", text[start.end():], re.M)
    body = text[start.end():start.end() + following.start()] if following else text[start.end():]
    commands = [line for line in body.replace("\\\n", " ").splitlines() if line.startswith("CMD ")]
    assert len(commands) == 1, f"the {stage} stage has {len(commands)} CMD lines"
    return commands[0]


# --------------------------------------------------------------------------- the default stack


def test_the_default_stack_builds_the_production_stage(base):
    build = base["services"]["backend"]["build"]
    assert build["target"] == "production"
    assert build["context"] == _SOURCE
    assert "args" not in build, "one API dependency set for every build, Leiden included (PRD-262 US-027)"


def test_nothing_of_the_checkout_is_mounted_over_the_image(base):
    mounts = _mounts(base["services"]["backend"])
    over_code = {target: source for target, (source, _) in mounts.items()
                 if source.startswith(_SOURCE) or target == "/app" or target in _SCRIPTS}
    assert not over_code, f"docker-compose.yml lays the checkout over the image: {over_code}"


def test_testers_mounts_stay_on_the_production_backend(base):
    """The four TESTER needs (PRD-262 §9-Q1): data, logs and both /workspaces mounts."""
    mounts = _mounts(base["services"]["backend"])
    assert mounts["/app/data"] == ("backend_data", "rw")
    assert mounts["/app/logs"] == ("backend_logs", "rw")
    assert base["volumes"]["backend_logs"]["name"] == "automatos_backend_logs"
    assert mounts["/workspaces"] == ("workspaces_data", "ro")
    assert mounts[_LOCAL_ROOT][0].startswith("${AUTOMATOS_WORKSPACE_DIR") and mounts[_LOCAL_ROOT][1] == "ro"


def test_the_production_backend_keeps_both_env_files_with_api_local_last(base):
    """next-cycle.sh edits envs/api.local, then `docker compose up -d --no-deps backend`."""
    assert base["services"]["backend"]["env_file"] == [
        "envs/api.defaults",
        {"path": "envs/api.local", "required": False},
    ]


def test_the_image_command_runs_and_web_concurrency_sets_the_workers(base):
    """No compose command or entrypoint: the production CMD starts uvicorn with no worker
    flag, so WEB_CONCURRENCY=1 from envs/api.defaults sets the count (PRD-262 US-029)."""
    backend = base["services"]["backend"]
    assert "command" not in backend and "entrypoint" not in backend
    assert "WEB_CONCURRENCY" not in (backend.get("environment") or {})
    assert "WEB_CONCURRENCY=1" in _API_DEFAULTS.read_text(encoding="utf-8").splitlines()
    production = _stage_cmd("production")
    assert "--reload" not in production and "--workers" not in production, production


# --------------------------------------------------------------------------- make dev


def test_make_dev_keeps_the_development_stage_and_its_reload(dev):
    assert dev["services"]["backend"]["build"]["target"] == "development"
    assert dev["services"]["frontend"]["build"]["target"] == "development"
    assert "--reload" in _stage_cmd("development")


def test_make_dev_mounts_the_source_and_takes_over_the_script_mounts(dev):
    mounts = _mounts(dev["services"]["backend"])
    assert mounts["/app"] == (_SOURCE, "rw")
    for script in _SCRIPTS:
        assert mounts[script] == (f"{_SOURCE}/{script.rsplit('/', 1)[1]}", "ro")


def test_make_dev_changes_the_stage_and_the_mounts_only(base, dev):
    """The env files, the command, the health check and TESTER's mounts come from docker-compose.yml."""
    backend = dev["services"]["backend"]
    overridden = {"env_file", "environment", "command", "entrypoint", "healthcheck", "depends_on"} & set(backend)
    assert not overridden, f"docker-compose.dev.yml overrides the backend's {sorted(overridden)}"
    assert "args" not in backend["build"], "make dev builds with docker-compose.yml's build args"
    dev_mounts, base_mounts = _mounts(backend), _mounts(base["services"]["backend"])
    for target in set(dev_mounts) & set(base_mounts):
        assert dev_mounts[target] == base_mounts[target], target


def test_make_dev_runs_the_override():
    makefile = (_REPO / "Makefile").read_text(encoding="utf-8")
    assert re.search(rf"^DEV_COMPOSE\s*\?=\s*{re.escape(_DEV_COMMAND)}$", makefile, re.M)
    recipe = makefile.split("\ndev:\n", 1)[1].split("\n\n", 1)[0]
    assert "$(DEV_COMPOSE) up -d --build" in recipe


# --------------------------------------------------------------------------- the docs


def _section(path: str, heading: str) -> str:
    text = (_REPO / path).read_text(encoding="utf-8")
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def test_the_dev_override_says_what_each_stack_runs():
    header = " ".join((_REPO / "docker-compose.dev.yml").read_text(encoding="utf-8")
                      .split("\nservices:", 1)[0].replace("#", " ").split())
    assert "runs the published production images, pulled from GHCR" in header
    assert "make dev" in header and "--reload" in header


@pytest.mark.parametrize("path,heading", [
    ("AGENTS.md", "## Running things"),
    ("CONTRIBUTING.md", "## Development environment"),
])
def test_the_contributor_docs_name_make_dev_as_the_development_environment(path, heading):
    section = _section(path, heading)
    assert "`make dev`" in section and _DEV_COMMAND in section
    assert "Plain `docker compose up` and `make up` run the published images, not your checkout" in section


def test_contributing_no_longer_says_the_default_stack_reloads():
    section = _section("CONTRIBUTING.md", "## Development environment")
    assert "The backend and frontend containers bind-mount" not in section
    assert "Migrations.** Database migrations run once per boot, from the backend's entrypoint" in section


def test_the_self_hosting_guide_says_the_default_backend_is_the_production_stage():
    guide = (_REPO / "docs" / "getting-started" / "self-hosting.md").read_text(encoding="utf-8")
    backend_row = next(line for line in guide.splitlines() if line.startswith("| `backend` |"))
    assert "(`production` target)" in backend_row and "`make dev`" in backend_row
    working = " ".join(_section("docs/getting-started/self-hosting.md",
                                "### Working on the code instead of with it").split())
    assert "runs the published production images, pulled from GHCR" in working
    assert "make dev" in working and "--reload" in working
