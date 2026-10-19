"""PRD-262 US-017: local builds stop baking uploads and caches into the image.

A self-hoster's checkout collects runtime files. The app writes generated documents,
thumbnails and brand files under DOCUMENT_STORAGE_DIR, whose code default `documents`
is /app/documents, and the compose bind mount put that in ./orchestrator. Nested
__pycache__ and the eval and import-linter caches sit there too. Every source build sent
them to Docker, and the development stage stored them twice (`COPY . .`, `chown -R`).

This suite checks:

- orchestrator/.dockerignore, read with Docker's pattern rules (a pattern matches from
  the context root, `**` crosses folders, a match on a folder covers what is in it): it
  leaves out DOCUMENT_STORAGE_DIR's default, nested bytecode and the caches, keeps
  modules/documents and the skill seeds, and drops no tracked file;
- envs/api.defaults puts the documents on the backend_data volume, which every compose
  file mounts at /app/data;
- the development stage copies no code, stores nothing twice, and still makes the
  directories the volumes mount on; a compose file that builds it mounts the source;
- scripts/ci/build_context.py: its seeds land where those rules apply, its budget holds
  today's code, and it reads the context transfer off real BuildKit logs;
- CI seeds before the builds it checks: test.yml's prod-image-parity (the production
  build) and smoke-fresh-clone (make build since PRD-262 US-039), whose script fails over
  the budget once make build has exited.

File reads and one `git ls-files`; the smoke script runs against a fake docker and make.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "orchestrator"
DOCKERIGNORE = (ORCHESTRATOR / ".dockerignore").read_text(encoding="utf-8")
DOCKERFILE = (ORCHESTRATOR / "Dockerfile").read_text(encoding="utf-8")
API_DEFAULTS = (ROOT / "envs" / "api.defaults").read_text(encoding="utf-8")
SMOKE_SCRIPT = ROOT / "scripts" / "ci" / "smoke-fresh-clone.sh"
WORKFLOWS = ROOT / ".github" / "workflows"
# docker-compose.images.yml went with PRD-262 US-036: docker-compose.yml names the published images.
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.dev.yml")
WORKDIR = "/app"
US017_RULES = (
    "**/__pycache__",
    "**/*.py[cod]",
    "/documents",
    "scripts/eval/tool_routing/.embedding_cache.json",
    "scripts/eval/**/results/",
    ".import_linter_cache",
)
WRITABLE = ("/app/logs", "/app/vector_stores", "/app/projects", "/app/exports", "/app/data")
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_CHOWN_R = re.compile(r"\bchown\s+(?:-\S*R|--recursive)")
_NAMED_VOLUME = re.compile(r"[A-Za-z0-9][\w.-]*")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


context = _load("build_context")


# --------------------------------------------------------------------------- Docker's rules


def _docker_regex(pattern: str) -> re.Pattern:
    """moby/patternmatcher's translation of one cleaned pattern into a regular expression."""
    out, i = "^", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            i += 2
            if pattern[i:i + 1] == "/":
                i += 1
            out += ".*" if i >= len(pattern) else "(.*/)?"
            continue
        char = pattern[i]
        if char == "*":
            out += "[^/]*"
        elif char == "?":
            out += "[^/]"
        elif char in ".+()|{}$":
            out += "\\" + char
        else:
            out += char
        i += 1
    return re.compile(out + "$")


def _rules(text: str, without: tuple = ()) -> list:
    """(re-include?, regex) per line, cleaned as Docker reads a .dockerignore."""
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line in without:
            continue
        keep = line.startswith("!")
        line = os.path.normpath(line[1:].strip() if keep else line)
        if len(line) > 1 and line.startswith("/"):
            line = line[1:]
        rules.append((keep, _docker_regex(line)))
    return rules


RULES = _rules(DOCKERIGNORE)


def ignored(path: str, rules: list = RULES) -> bool:
    """Whether the build leaves `path` (relative to the context) out: the last rule that
    matches it or one of its folders decides."""
    parts = path.split("/")
    prefixes = ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
    result = False
    for keep, regex in rules:
        if any(regex.match(prefix) for prefix in prefixes):
            result = not keep
    return result


@functools.lru_cache(maxsize=1)
def _tracked() -> tuple:
    """(path in the context, size) of every file git tracks under orchestrator/."""
    listing = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "orchestrator"],
                             capture_output=True, check=True).stdout.decode("utf-8")
    files = []
    for path in filter(None, listing.split("\0")):
        if (ROOT / path).is_file():
            files.append((path.removeprefix("orchestrator/"), (ROOT / path).stat().st_size))
    assert files, "git lists no file under orchestrator/"
    return tuple(files)


@pytest.mark.parametrize("rule,path,left_out", [
    ("*.md", "README.md", True),
    ("*.md", "core/seeds/platform-management-skill.md", False),  # root only
    ("__pycache__/", "core/__pycache__/config.cpython-311.pyc", False),  # the old rule
    ("**/__pycache__", "core/__pycache__/config.cpython-311.pyc", True),
    ("/documents", "modules/documents/generation_service.py", False),
    ("**/documents", "modules/documents/generation_service.py", True),
    ("scripts/eval/**/results/", "scripts/eval/results/run.json", True),  # ** may be no folder
])
def test_the_rules_are_read_the_way_docker_reads_them(rule, path, left_out):
    assert ignored(path, _rules(rule)) is left_out


# --------------------------------------------------------------------------- .dockerignore


def test_us017s_rules_replace_the_root_only_ones():
    lines = [line.strip() for line in DOCKERIGNORE.splitlines()]
    assert set(US017_RULES) <= set(lines)
    for gone in ("__pycache__/", "*.py[cod]"):
        assert gone not in lines, f"{gone} matches the context root only; the **/ rule replaces it"
    for never in ("**/*.md", "**/documents"):
        assert never not in lines, f"{never} drops code the app loads"


def _config_default(name: str) -> str:
    """The literal default config.py passes to os.getenv for `name`."""
    tree = ast.parse((ORCHESTRATOR / "config.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "getenv" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == name):
            return node.args[1].value
    raise AssertionError(f"config.py reads no {name} with a default")


def test_the_document_storage_default_is_left_out_of_the_context():
    """The default is relative, so it resolves under the WORKDIR, /app: the context's root."""
    default = _config_default("DOCUMENT_STORAGE_DIR")
    in_context = posixpath.relpath(posixpath.join(WORKDIR, default), WORKDIR)
    assert not in_context.startswith(".."), f"{default} is outside /app, so outside the context"
    assert ignored(in_context)
    assert ignored(f"{in_context}/00000000-0000-0000-0000-0000000000c1/generated/report.docx")
    assert ignored(f"{in_context}/00000000-0000-0000-0000-0000000000c1/brand/logo.png")


@pytest.mark.parametrize("path", [
    "__pycache__/main.cpython-311.pyc",
    "core/__pycache__/config.cpython-311.pyc",
    "modules/documents/__pycache__/generation_service.cpython-311.pyc",
    "core/stray.pyc",
    "modules/agents/stray.pyo",
    "scripts/eval/tool_routing/.embedding_cache.json",
    "scripts/eval/tool_routing/results/2026-10-18.json",
    "scripts/eval/memory_recall/results/run/scores.json",
    ".import_linter_cache/cache.json",
])
def test_nested_bytecode_and_the_caches_are_left_out(path):
    assert ignored(path)


def test_us017s_rules_drop_no_tracked_file():
    """They name runtime files only: every tracked file the build sent before, it still sends."""
    before = _rules(DOCKERIGNORE, without=US017_RULES)
    dropped = [path for path, _ in _tracked() if ignored(path) and not ignored(path, before)]
    assert not dropped, dropped


def test_the_code_the_app_loads_stays_in_the_context():
    paths = [path for path, _ in _tracked()]
    documents = [path for path in paths if path.startswith("modules/documents/")]
    seeds = [path for path in paths if path.startswith("core/seeds/") and path.endswith(".md")]
    assert documents and seeds
    assert not [path for path in documents + seeds if ignored(path)]
    # pybuild copies the image lock and the splitter (PRD-262 US-022; one lock since US-027).
    for path in ("main.py", "config.py", "alembic.ini", "requirements.lock",
                 "scripts/split_lock.py", "docker-entrypoint.sh"):
        assert not ignored(path), path


# --------------------------------------------------------------------------- the data volume


def _compose(name: str) -> dict:
    class Loader(yaml.SafeLoader):
        """Reads compose's !reset and !override tags as their plain value."""

    def plain(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("!", plain)
    return yaml.load((ROOT / name).read_text(encoding="utf-8"), Loader=Loader)


def _backend_mounts(name: str) -> list:
    """(source, target) of each volume the compose file gives the backend."""
    mounts = []
    for entry in (_compose(name)["services"].get("backend") or {}).get("volumes") or []:
        if isinstance(entry, dict):
            mounts.append((entry.get("source", ""), entry.get("target", "")))
        else:
            mounts.append(tuple((entry.split(":") + [""])[:2]))
    return mounts


def test_the_stack_keeps_generated_documents_on_the_data_volume():
    match = re.search(r"^DOCUMENT_STORAGE_DIR=(\S+)$", API_DEFAULTS, re.M)
    assert match and match.group(1) == "/app/data/documents"
    storage = match.group(1)
    for name in COMPOSE_FILES:
        volumes = [target for source, target in _backend_mounts(name) if _NAMED_VOLUME.fullmatch(source)]
        assert any(storage.startswith(target.rstrip("/") + "/") for target in volumes), (
            f"{name}: no named volume holds {storage}")


# --------------------------------------------------------------------------- the development stage


def _instructions(text: str) -> list:
    """The Dockerfile's instructions: continuation lines joined, comments dropped, spaces collapsed."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _stage(name: str) -> list:
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.fullmatch(rf"FROM \S+ as {name}", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start + 1:end]


DEVELOPMENT = _stage("development")
PRODUCTION = _stage("production")


def test_the_development_stage_copies_no_code_and_stores_nothing_twice():
    copies = [line for line in DEVELOPMENT if line.split()[0].upper() in ("COPY", "ADD")]
    assert [line.split()[-1] for line in copies] == [
        "/usr/local/bin/docker-entrypoint.sh", "/usr/local/bin/docker-healthcheck.sh",
        "/usr/local/bin/docker-secrets-guard.sh"]
    assert not [line for line in DEVELOPMENT if _CHOWN_R.search(line)]


def test_the_development_stage_makes_the_writable_directories_in_the_production_layer():
    """The same RUN as the production stage's first, so a build makes the layer once."""
    assert DEVELOPMENT[0] == PRODUCTION[0]
    commands = [command.split() for command in DEVELOPMENT[0].removeprefix("RUN ").split("&&")]
    assert commands[0] == ["useradd", "-m", "-u", "1000", "automatos"]
    mkdir = next(command for command in commands if command[0] == "mkdir")
    chown = next(command for command in commands if command[0] == "chown")
    assert set(WRITABLE) <= set(mkdir[2:])
    assert chown[1] == "automatos:automatos" and set(WRITABLE) <= set(chown[2:])
    assert "USER automatos" in DEVELOPMENT


def test_every_compose_file_that_builds_the_development_stage_mounts_the_source():
    building = []
    for name in COMPOSE_FILES:
        build = (_compose(name)["services"].get("backend") or {}).get("build")
        if isinstance(build, dict) and build.get("target") == "development":
            building.append(name)
            assert ("./orchestrator", "/app") in _backend_mounts(name), f"{name} runs the stage without code"
    assert building, "no compose file builds the development stage"


# --------------------------------------------------------------------------- build_context.py


def test_each_seed_is_left_out_and_breaks_the_budget_on_its_own():
    assert context.SEED_BYTES >= context.BUDGET_BYTES
    assert all(ignored(seed) for seed in context.SEEDS)
    default = _config_default("DOCUMENT_STORAGE_DIR")
    assert any(seed.startswith(f"{default}/") for seed in context.SEEDS)
    for rule in US017_RULES:
        assert any(ignored(seed, _rules(rule)) for seed in context.SEEDS), f"no seed tries {rule}"


def test_todays_code_fits_the_budget():
    sent = sum(size for path, size in _tracked() if not ignored(path))
    assert 0 < sent < context.BUDGET_BYTES, (
        f"the API's code is {sent / 1e6:.2f} MB: raise BUDGET_BYTES in scripts/ci/build_context.py "
        "(it stays below every seed, which is as large as the budget)")


def test_seed_writes_each_missing_file_and_leaves_the_rest(tmp_path):
    kept = tmp_path / context.SEEDS[3]
    kept.parent.mkdir(parents=True)
    kept.write_text("a real cache", encoding="utf-8")
    written = context.seed(tmp_path, size=3000)
    assert sorted(written) == sorted(tmp_path / seed for seed in context.SEEDS if seed != context.SEEDS[3])
    assert all(path.stat().st_size == 3000 for path in written)
    assert kept.read_text(encoding="utf-8") == "a real cache"
    assert context.seed(tmp_path, size=3000) == []


# From the 18 Oct smoke-fresh-clone run (compose, three builds in one session), with the
# backend's transfer split as BuildKit prints a long one, and a RUN line that only quotes one.
COMPOSE_LOG = """\
#0 building with "default" instance using docker driver

#12 [backend internal] load .dockerignore
#12 transferring context: 1.10kB done
#12 DONE 0.0s

#16 [workspace-worker internal] load build context
#16 transferring context: 188.75kB done
#16 DONE 0.1s

#19 [backend internal] load build context
#19 transferring context: 6.29MB 0.5s

#20 [frontend internal] load build context
#20 transferring context: 28.32MB 1.3s done
#20 DONE 1.4s

#19 [backend internal] load build context
#19 transferring context: 19.81MB 1.1s done
#19 DONE 1.2s

#23 [backend base 3/3] RUN echo
#23 0.512 transferring context: 99MB
#23 DONE 0.6s
"""
# From the same day's prod-image-parity job log, as GitHub serves it (time-stamped).
DOCKER_LOG = """\
2026-10-18T20:28:35.8151401Z #4 [internal] load .dockerignore
2026-10-18T20:28:35.8155978Z #4 transferring context: 1.10kB done
2026-10-18T20:28:36.1553476Z #6 [internal] load build context
2026-10-18T20:28:36.1554032Z #6 transferring context: 19.81MB 0.4s done
"""


@pytest.mark.parametrize("log,service,sent", [
    (COMPOSE_LOG, "backend", 19_810_000),
    (COMPOSE_LOG, "frontend", 28_320_000),
    (COMPOSE_LOG, "workspace-worker", 188_750),
    (DOCKER_LOG, None, 19_810_000),
])
def test_check_reads_what_one_build_sent_as_its_context(log, service, sent):
    assert sum(context.transfers(log, service).values()) == sent


@pytest.mark.parametrize("shown,size", [
    ("2B", 2), ("32.77kB", 32_770), ("19.81MB", 19_810_000), ("1.5GB", 1_500_000_000), ("3MiB", 3 << 20),
])
def test_check_reads_buildkits_units(shown, size):
    log = f"#6 [internal] load build context\n#6 transferring context: {shown} 0.1s done\n"
    assert context.transfers(log, None) == {"6": size}


def test_check_fails_a_leaked_upload_and_a_log_without_the_transfer():
    leaked = DOCKER_LOG.replace("19.81MB 0.4s done", "59.81MB 2.0s done")
    passed, message = context.verdict(leaked, None)
    assert not passed and "59.81 MB" in message and "orchestrator/.dockerignore" in message
    assert context.verdict(DOCKER_LOG, None) == (True, "`[internal] load build context` sent 19.81 MB, "
                                                       "within the 40.00 MB budget")
    passed, message = context.verdict(COMPOSE_LOG, None)
    assert not passed and "no `transferring context` line" in message


def test_check_exits_1_with_an_annotation_over_the_budget(tmp_path, capsys):
    log = tmp_path / "build.log"
    log.write_text(COMPOSE_LOG, encoding="utf-8")
    assert context.main(["check", str(log), "--service", "backend"]) == 0
    assert "sent 19.81 MB, within" in capsys.readouterr().out
    assert context.main(["check", str(log), "--service", "frontend"]) == 0
    assert "sent 28.32 MB, within" in capsys.readouterr().out
    log.write_text(COMPOSE_LOG.replace("19.81MB 1.1s", "61.2MB 3.1s"), encoding="utf-8")
    assert context.main(["check", str(log), "--service", "backend"]) == 1
    assert capsys.readouterr().out.startswith("::error::")


# --------------------------------------------------------------------------- CI


def _steps(workflow: str, job: str) -> list:
    return yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"][job]["steps"]


def _at(runs: list, needle: str) -> int:
    found = [i for i, run in enumerate(runs) if needle in run]
    assert len(found) == 1, f"{needle!r} in {len(found)} steps"
    return found[0]


def test_prod_image_parity_seeds_builds_and_reads_the_context():
    runs = [step.get("run", "") for step in _steps("test.yml", "prod-image-parity")]
    seeded = _at(runs, "python3 scripts/ci/build_context.py seed orchestrator")
    built = _at(runs, "-t parity-be orchestrator/")
    checked = _at(runs, "python3 scripts/ci/build_context.py check")
    assert seeded < built < checked
    build = runs[built]
    log = re.search(r'-t parity-be orchestrator/ 2>&1 \| tee "(\$RUNNER_TEMP/[\w.-]+)"', build).group(1)
    assert build.index("set -o pipefail") < build.index("docker build"), "tee would hide a failed build"
    assert runs[checked].strip() == f'python3 scripts/ci/build_context.py check "{log}"'
    changes = next(step for step in _steps("test.yml", "changes") if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    for path in ("scripts/ci/build_context.py", "orchestrator/.dockerignore"):
        assert re.search(changes["env"]["PROD_IMAGE_PATHS"], path), path


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_fresh_clone_smoke_seeds_and_reads_the_backends_build():
    runs = [step.get("run", "") for step in _steps("smoke-fresh-clone.yml", "fresh-clone-boot")]
    assert _at(runs, "python3 scripts/ci/build_context.py seed orchestrator") < _at(
        runs, "bash scripts/ci/smoke-fresh-clone.sh")
    workflow = yaml.safe_load((WORKFLOWS / "smoke-fresh-clone.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert {"orchestrator/.dockerignore", "scripts/ci/build_context.py"} <= set(triggers[event]["paths"])
    script = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"^set -uo pipefail$", script, re.M), "tee would hide a failed build"
    assert 'CONTEXT_TOOL="scripts/ci/build_context.py"' in script
    assert 'BUILD_LOG="${OUT_DIR}/build.log"' in script
    # PRD-262 US-039: the build is make build's, its output the build log; once make build
    # has exited, the log is whole and the context is read off it.
    build = _function(script, "build_images")
    assert 'make build 2>&1 | tee "$BUILD_LOG"' in build
    finish = _function(script, "finish_make")
    assert finish.rstrip().endswith('[ "$MAKE_TARGET" != "build" ] || check_build_context')
    assert 'python3 "$CONTEXT_TOOL" check --service backend "$BUILD_LOG"' in _function(script, "check_build_context")


# A docker that starts nothing, and a make build whose backend build sent a context over
# the budget and then exited 0.
_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
exit 0
"""
_FAKE_MAKE = """#!/bin/bash
echo "make $*" >> "$CALL_LOG"
printf '#19 [backend internal] load build context\\n#19 transferring context: 61.20MB 3.1s done\\n' >&2
exit 0
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_smoke_fails_when_make_builds_context_is_over_budget(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {"docker": _FAKE_DOCKER, "make": _FAKE_MAKE, "python3": f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'}
    for name, body in stubs.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    calls = tmp_path / "calls.log"
    calls.touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(calls),
           "SMOKE_OUT_DIR": str(tmp_path / "out"), "RUNNER_TEMP": str(tmp_path),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}
    proc = subprocess.run(["bash", str(SMOKE_SCRIPT)], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    out, called = proc.stdout + proc.stderr, calls.read_text(encoding="utf-8").splitlines()
    assert proc.returncode == 1, out
    assert "sent 61.20 MB, over the 40.00 MB budget" in out
    assert "61.20MB" in (tmp_path / "out" / "build.log").read_text(encoding="utf-8")
    assert "make build" in called
    assert not any(" up " in call for call in called if call.startswith("docker ")), "the smoke starts nothing"
    assert called[-1] == "docker compose -f docker-compose.yml down -v"
    assert "build context is over its budget" in (tmp_path / "summary.md").read_text(encoding="utf-8")
