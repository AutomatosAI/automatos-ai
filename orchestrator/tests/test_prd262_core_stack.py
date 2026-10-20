"""PRD-262 US-040: a five-container core stack.

``docker-compose.yml`` keeps the workspace worker in the default stack with no profile, so
the default stays six containers for every ``.env``, old or new. ``docker-compose.core.yml``
(``make up CORE=1``) puts the worker behind a ``workspace`` profile and runs the core five:
postgres, redis, minio, backend and frontend. This suite checks, from the files alone:

* the default six and the override's five; the backend's dependency on the worker is
  ``required: false`` while the datastores stay required;
* the override mounts the workspaces volume writable in the backend, in the place of the
  read-only mount: Docker must create the deliverables folder's mount point inside that
  volume, which it cannot do on a read-only mount of a fresh volume;
* both backend stages create ``/workspaces`` owned by the automatos user (uid 1000) before
  ``USER``, so a fresh volume is uid 1000 whichever service mounts it first, and
  prod-image-parity reads that owner inside the built image;
* the Makefile: ``CORE=1`` exports ``COMPOSE_FILE`` with the override, and ``make up`` and
  ``make build`` first remove a worker an earlier six-container run started;
* ``scripts/stack_images.py``: a ``COMPOSE_FILE`` naming the override leaves the worker's
  image alone (not pulled, re-tagged or built), unless ``COMPOSE_PROFILES`` turns its
  profile on;
* the guide's §1b gives the commands, and smoke-images-pr runs
  ``scripts/ci/smoke-core-stack.sh`` on the images it built.

Pure file reads and the stack_images functions with a fake Docker and registry: no Docker,
no network. The smoke itself runs only in CI.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.workspace_client import WORKER_NOT_RUNNING

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("stack_images_us040", _REPO / "scripts" / "stack_images.py")
stack = importlib.util.module_from_spec(_SPEC)
sys.modules["stack_images_us040"] = stack  # before exec: dataclasses look their module up
_SPEC.loader.exec_module(stack)

MAKEFILE = (_REPO / "Makefile").read_text(encoding="utf-8")
SCRIPT_PATH = _REPO / "scripts" / "ci" / "smoke-core-stack.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
DOCKERFILE = (_REPO / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
GUIDE = (_REPO / "docs" / "getting-started" / "self-hosting.md").read_text(encoding="utf-8")

DATASTORES = {"postgres", "redis", "minio"}
FIVE = DATASTORES | {"backend", "frontend"}
SIX = FIVE | {"workspace-worker"}
CORE_FILES = "docker-compose.yml:docker-compose.core.yml"
LOCAL_ROOT = "/workspaces/${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"
_CONTINUATION = re.compile(r"\\[ \t]*\n")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that reads compose's !reset and !override tags as plain values."""


def _untagged(loader, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


for _tag in ("!reset", "!override"):
    _ComposeLoader.add_constructor(_tag, _untagged)


def _compose(name: str) -> dict:
    return yaml.load((_REPO / name).read_text(encoding="utf-8"), Loader=_ComposeLoader)


BASE = _compose("docker-compose.yml")
CORE = _compose("docker-compose.core.yml")


def _one_shots(services: dict) -> set:
    """Services another one waits on to finish (service_completed_successfully): they run and exit."""
    waited = set()
    for body in services.values():
        depends = (body or {}).get("depends_on") or {}
        if isinstance(depends, dict):
            waited |= {name for name, dep in depends.items()
                       if isinstance(dep, dict) and dep.get("condition") == "service_completed_successfully"}
    return waited


def _running(services: dict) -> set:
    """The long-running services plain `up` starts: those with no profile, less the one-shots.

    minio-volume-owner (PRD-262 US-041) has no profile but exits before MinIO starts.
    """
    return {name for name, body in services.items() if not (body or {}).get("profiles")} - _one_shots(services)


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


def _mounts(volumes: list) -> dict:
    """``{target: (source, read_only)}`` for short- and long-syntax volume entries."""
    out = {}
    for entry in volumes:
        if isinstance(entry, dict):
            out[entry["target"]] = (entry.get("source", ""), bool(entry.get("read_only", False)))
        else:
            source, target, *mode = _fields(entry)
            out[target] = (source, "ro" in (mode[0].split(",") if mode else []))
    return out


# --------------------------------------------------------------------------- compose


def test_the_default_stack_is_six_with_the_worker_and_no_env_key_turns_it_off():
    assert _running(BASE["services"]) == SIX
    assert "profiles" not in BASE["services"]["workspace-worker"]
    keys = {line.split("=", 1)[0] for line in (_REPO / ".env.example").read_text(encoding="utf-8").splitlines()
            if re.match(r"^[A-Z_]+=", line)}
    assert not keys & {"COMPOSE_PROFILES", "COMPOSE_FILE"}, "the core stack is an override, not an .env key"


def test_the_core_override_runs_the_five_and_touches_only_the_worker_and_one_mount():
    assert set(CORE) == {"services"}
    assert set(CORE["services"]) == {"workspace-worker", "backend"}
    assert CORE["services"]["workspace-worker"] == {"profiles": [stack.WORKER_PROFILE]}
    assert set(CORE["services"]["backend"]) == {"volumes"}
    merged = {name: {**(body or {}), **(CORE["services"].get(name) or {})} for name, body in BASE["services"].items()}
    assert _running(merged) == FIVE


def test_the_backend_starts_without_the_worker_and_still_waits_for_the_datastores():
    depends = BASE["services"]["backend"]["depends_on"]
    assert depends["workspace-worker"] == {"condition": "service_started", "required": False}
    for name in DATASTORES:
        assert depends[name] == {"condition": "service_healthy"}, f"{name} stays a required dependency"


def test_the_core_backend_mounts_the_workspaces_volume_writable_in_place_of_the_read_only_one():
    """Compose merges a service's volumes by target, so the override's entry replaces the base's."""
    base = _mounts(BASE["services"]["backend"]["volumes"])
    entries = CORE["services"]["backend"]["volumes"]
    assert base["/workspaces"] == ("workspaces_data", True)
    assert entries == [{"type": "volume", "source": "workspaces_data", "target": "/workspaces", "read_only": False}], (
        "long syntax with read_only: false spelled out, so no merge can keep the base's read-only flag")
    merged = {**base, **_mounts(entries)}
    assert merged["/workspaces"] == ("workspaces_data", False)
    assert LOCAL_ROOT in merged and merged[LOCAL_ROOT][1] is True, "your deliverables folder stays read-only"
    assert LOCAL_ROOT.startswith("/workspaces/"), "its mount point lies inside the volume"
    assert {"/app/data", "/app/logs"} <= set(merged), "the backend keeps its own volumes (PRD-262 §9-Q1)"


# --------------------------------------------------------------------------- the backend image


def _instructions(text: str) -> list:
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _stage(name: str) -> list:
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.fullmatch(rf"FROM \S+ as {name}", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start + 1:end]


@pytest.mark.parametrize("stage", ["development", "production"])
def test_both_backend_stages_make_workspaces_the_users_before_user(stage):
    lines = _stage(stage)
    user_at = lines.index("USER automatos")
    run_at = next(i for i, line in enumerate(lines) if line.startswith("RUN ") and "/workspaces" in line.split())
    commands = [command.split() for command in lines[run_at].removeprefix("RUN ").split("&&")]
    assert run_at < user_at, "created as root, before the image switches to its user"
    assert ["useradd", "-m", "-u", "1000", "automatos"] in commands, "the owner is uid 1000"
    mkdir = next(command for command in commands if command[0] == "mkdir")
    chown = next(command for command in commands if command[0] == "chown")
    assert mkdir[1] == "-p" and "/workspaces" in mkdir[2:]
    assert chown[1] in ("automatos:automatos", "1000:1000") and "/workspaces" in chown[2:], "no -R: it is empty"
    assert commands.index(mkdir) < commands.index(chown)


def test_ci_reads_the_owner_of_workspaces_inside_the_built_image():
    workflow = yaml.safe_load((_REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "stat parity-be -c '%u:%g' /workspaces" in step.get("run", ""))
    assert built < check
    assert '"$owner" != "1000:1000"' in steps[check]["run"]


# --------------------------------------------------------------------------- the Makefile


def _recipe(target: str) -> list:
    body = MAKEFILE.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
    return [line.strip().lstrip("-@") for line in body.splitlines() if line.startswith("\t")]


def test_core_hands_every_compose_command_the_override_through_compose_file():
    assert re.search(rf"^CORE_COMPOSE_FILE := {re.escape(CORE_FILES)}$", MAKEFILE, re.M)
    assert "ifeq ($(CORE),1)\nexport COMPOSE_FILE := $(CORE_COMPOSE_FILE)\nendif" in MAKEFILE
    assert re.search(r"^COMPOSE\s*\?= docker compose$", MAKEFILE, re.M), "no -f, so COMPOSE_FILE applies"
    assert CORE_FILES.split(":")[1] == stack.CORE_OVERRIDE, "the script looks for the file make names"


def test_make_up_and_build_remove_a_worker_an_earlier_run_started_before_the_stack_starts():
    define = re.search(r"^define remove_core_worker\n(.*?)\nendef$", MAKEFILE, re.M | re.S).group(1)
    assert define == "$(if $(filter 1,$(CORE)),@$(COMPOSE) -f docker-compose.yml rm --stop --force workspace-worker)"
    for target in ("up", "build"):
        recipe = _recipe(target)
        at = recipe.index("$(remove_core_worker)")
        assert recipe[at + 1] == "$(COMPOSE) up -d --no-build --remove-orphans", target


def test_the_guide_gives_the_core_commands():
    start = GUIDE.index("## 1b. ")
    section = GUIDE[start:GUIDE.index("\n## ", start + 1)]
    for command in ("make up CORE=1", f"export COMPOSE_FILE={CORE_FILES}",
                    "docker compose -f docker-compose.yml rm --stop --force workspace-worker",
                    "docker compose -f docker-compose.yml -f docker-compose.core.yml up -d"):
        assert command in section, command
    assert "docker compose -f docker-compose.yml -f docker-compose.core.yml up -d" in (
        _REPO / "docker-compose.yml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- make up's images


TWO = ("backend", "frontend")
HEAD = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
LOCAL = "local-0d9f0ac"
TREES = {"orchestrator": "1" * 40, "frontend": "2" * 40, "services/workspace-worker": "3" * 40}


def _tree(service: str) -> str:
    app = stack.app_for(service)
    return stack.image_ref(app, "tree-" + TREES[app.context][:12])


@pytest.mark.parametrize("environ, dotenv, expected", [
    ({}, {}, stack.SERVICES),
    ({"COMPOSE_FILE": CORE_FILES}, {}, TWO),
    ({}, {"COMPOSE_FILE": CORE_FILES}, TWO),
    ({"COMPOSE_FILE": "docker-compose.yml"}, {"COMPOSE_FILE": CORE_FILES}, stack.SERVICES),
    ({"COMPOSE_FILE": "/srv/automatos/docker-compose.yml:/srv/automatos/docker-compose.core.yml"}, {}, TWO),
    ({"COMPOSE_FILE": "docker-compose.yml;docker-compose.core.yml", "COMPOSE_PATH_SEPARATOR": ";"}, {}, TWO),
    ({"COMPOSE_FILE": CORE_FILES, "COMPOSE_PROFILES": "media"}, {}, TWO),
    ({"COMPOSE_FILE": CORE_FILES, "COMPOSE_PROFILES": "media,workspace"}, {}, stack.SERVICES),
    ({"COMPOSE_FILE": CORE_FILES, "COMPOSE_PROFILES": "*"}, {}, stack.SERVICES),
])
def test_compose_file_naming_the_override_leaves_the_worker_out(environ, dotenv, expected):
    if os.pathsep != ":" and "COMPOSE_PATH_SEPARATOR" not in environ:
        pytest.skip("compose splits COMPOSE_FILE on ':' only on Linux and macOS")
    assert stack.stack_services(dotenv, environ) == expected


def test_only_drops_the_workers_pull_re_tag_and_build():
    plan = stack.Plan(LOCAL, "why", retag=(("backend", _tree("backend")), ("workspace-worker", _tree("workspace-worker"))),
                      build=(("frontend", "args"), ("workspace-worker", "dirty")))
    core = stack.only(plan, TWO)
    assert core.services == TWO
    assert core.retag == (("backend", _tree("backend")),) and core.build == (("frontend", "args"),)
    assert (core.tag, core.reason) == (plan.tag, plan.reason)
    assert stack.only(plan, stack.SERVICES) == plan


class FakeDocker:
    """Records the docker commands make up would run; ``fail`` names the ones that fail."""

    def __init__(self, local=(), fail=()):
        self._local, self._fail, self.calls = set(local), set(fail), []

    def local(self, image):
        return image in self._local

    def run(self, *args, env=None):
        self.calls.append(args)
        return 1 if args[:2] in self._fail else 0

    def check(self, *args, env=None):
        if self.run(*args, env=env) != 0:
            raise stack.Stop(f"docker {' '.join(args)} failed")


class FakeLookup:
    """The registry has every tree tag and nothing else."""

    def local(self, image):
        return False

    def published(self, image):
        return image in {_tree(service) for service in stack.SERVICES}

    def revision(self, image):
        return None


def test_edge_on_the_core_stack_pulls_two_and_runs_the_two_it_has(capsys):
    plan = stack.only(stack.Plan("edge", "why", pull=True), TWO)
    docker = FakeDocker()
    stack.execute(plan, docker, {})
    assert docker.calls == [("compose", "pull", *TWO)]
    edge = {stack.image_ref(stack.app_for(service), "edge") for service in TWO}
    stack.execute(plan, FakeDocker(local=edge, fail={("compose", "pull")}), {})
    assert "could not pull edge" in capsys.readouterr().err, "the worker's edge image is not needed"


def test_make_up_says_it_leaves_the_workers_image_alone():
    lines = stack.describe(stack.only(stack.Plan("sha-0d9f0ac", "reason"), TWO))
    assert f"  workspace-worker: not run ({stack.CORE_OVERRIDE}), so its image is left alone" in lines
    assert "  nothing is built" in lines
    assert not [line for line in stack.describe(stack.Plan("sha-0d9f0ac", "reason")) if "not run" in line]


def test_make_up_core_end_to_end_gets_no_worker_image(tmp_path, monkeypatch, capsys):
    """An unpublished commit re-tags every published tree; on the core stack, two of them."""
    (tmp_path / ".env").write_text(f"COMPOSE_FILE={CORE_FILES}\n", encoding="utf-8")
    docker = FakeDocker()
    monkeypatch.setattr(stack, "Docker", lambda: docker)
    monkeypatch.setattr(stack, "Registry", FakeLookup)
    monkeypatch.setattr(stack, "read_checkout", lambda root: stack.Checkout(HEAD, frozenset(), TREES))
    monkeypatch.setattr(stack.os, "environ", {})
    assert stack.main(["--root", str(tmp_path), "up"]) == 0
    assert docker.calls == [
        ("pull", _tree("backend")), ("tag", _tree("backend"), stack.image_ref(stack.app_for("backend"), LOCAL)),
        ("pull", _tree("frontend")), ("tag", _tree("frontend"), stack.image_ref(stack.app_for("frontend"), LOCAL)),
    ]
    assert (tmp_path / ".env").read_text(encoding="utf-8").endswith(f"AUTOMATOS_IMAGE_TAG={LOCAL}\n")
    assert "workspace-worker: not run" in capsys.readouterr().out


# --------------------------------------------------------------------------- CI


def _shell_value(name: str) -> str:
    return re.search(rf'^{name}="([^"]*)"$', SCRIPT, re.M).group(1)


def test_smoke_images_pr_runs_the_core_smoke_on_the_images_it_built():
    workflow = yaml.safe_load((_REPO / ".github" / "workflows" / "smoke-images.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["smoke-images-pr"]["steps"]
    boot = next(i for i, step in enumerate(steps) if step.get("run") == "bash scripts/ci/smoke-fresh-clone.sh")
    core = next(i for i, step in enumerate(steps) if step.get("run") == "bash scripts/ci/smoke-core-stack.sh")
    upload = next(i for i, step in enumerate(steps) if "upload-artifact" in step.get("uses", ""))
    assert boot < core < upload
    env = steps[core]["env"]
    assert env["AUTOMATOS_IMAGE_TAG"] == "${{ steps.target.outputs.tag }}"
    assert not {"POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY"} & set(env), "the .env files hold the secrets"
    assert env["SMOKE_OUT_DIR"].startswith(steps[upload]["with"]["path"].rstrip("/"))
    paths = workflow.get("on", workflow.get(True))["pull_request"]["paths"]
    assert {"scripts/ci/smoke-core-stack.sh", "docker-compose*.yml", "orchestrator/core/workspace_client.py",
            "orchestrator/api/workspace_files.py"} <= set(paths)


def test_the_smoke_expects_the_stacks_compose_defines_and_the_clients_message():
    names = {name: BASE["services"][name]["container_name"] for name in SIX}
    assert _shell_value("SIX") == " ".join(sorted(names.values()))
    assert _shell_value("FIVE") == " ".join(sorted(names[name] for name in FIVE))
    assert _one_shots(BASE["services"]) == {"minio-volume-owner"}
    assert _shell_value("CORE_SERVICES") == " ".join(sorted(FIVE | _one_shots(BASE["services"])))
    assert _shell_value("CORE_FILES") == CORE_FILES
    assert _shell_value("NOT_RUNNING") == WORKER_NOT_RUNNING


def test_the_smoke_runs_the_old_env_the_switch_and_a_fresh_core_stack_in_that_order():
    main = re.search(r"^main\(\) \{\n(.*?)^\}", SCRIPT, re.M | re.S).group(1).split()
    order = ["check_an_old_env_starts_six", "check_core_replaces_a_running_six", "teardown", "check_a_fresh_core_stack"]
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)
    fresh = re.search(r"^check_a_fresh_core_stack\(\) \{\n(.*?)^\}", SCRIPT, re.M | re.S).group(1)
    for step in ("plant_owner_probe", "check_core_config", "make up CORE=1", 'expect_running "$FIVE"',
                 "call_workspace_files 503", "up -d --no-build --wait", "call_workspace_files 200",
                 '[ "$owner" = "$probe_owner" ]', '[ "$owner" = "$folder_owner" ]'):
        assert step in fresh, step
    assert fresh.index("call_workspace_files 503") < fresh.index("up -d --no-build --wait"), "error first, worker after"


def test_the_old_env_is_the_one_users_had_before_w5():
    """The keys .env.example held before W5 (main at 0d9f0ac5), and no key W5's make writes."""
    body = re.search(r"cat > \"\$ENV_FILE\" <<EOF\n(.*?)\nEOF", SCRIPT, re.S).group(1)
    keys = [line.split("=", 1)[0] for line in body.splitlines()]
    assert keys == ["POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_PORT", "REDIS_PASSWORD",
                    "REDIS_PORT", "API_PORT", "API_KEY", "FRONTEND_PORT", "NEXT_PUBLIC_API_URL", "NEXT_PUBLIC_WS_URL",
                    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COMPOSIO_KEY", "FIRECRAWL_API_KEY",
                    "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY", "CLERK_SECRET_KEY", "CLERK_JWKS_URL",
                    "DEFAULT_WORKSPACE_ID", "REQUIRE_AUTH", "ENVIRONMENT", "LOG_LEVEL", "ADMINER_PORT"]
    for secret in ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY"):
        assert f"{secret}=$(openssl rand -hex 24)" in body, "filled, as the old QUICKSTART said"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_the_smoke_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT_PATH)], capture_output=True, timeout=30).returncode == 0
