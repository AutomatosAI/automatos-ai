"""PRD-262 US-022: pybuild installs the lock, exactly, in two layers, and CI installs it too.

orchestrator/Dockerfile:

- uv comes from a named stage pinned as ``tag@sha256`` (FR-9: Dependabot reads FROM lines),
  and only pybuild, which never ships, gets the binary;
- every build installs requirements.lock, Railway's included: no build arg picks a
  second lock (US-027 removed the graph-extras arg and the [leiden] lock), and no sed
  rewrites a requirement;
- the install is ``uv pip install --python /usr/local/bin/python --prefix /install/<part>
  --no-deps --require-hashes --no-compile-bytecode`` with US-015's binary flags, once per
  part of the lock (split_lock.py: the two halves, then the development stage's part,
  watchfiles, the 19 Oct breakdown's row #47), and a failed part fails the RUN. Run here
  with ``sh`` and a fake uv, it installs the parts of the lock, stable first;
- after US-020's strip and US-021's prune, site-packages is compiled per part with
  ``compileall -s <half> -p /usr/local --invalidation-mode unchecked-hash`` and a fixed
  PYTHONHASHSEED, never ending in ``|| true``. Run here on a fake tree, it writes
  unchecked-hash .pyc that name their /usr/local path, and fails on a file that does not
  compile;
- the base stage copies the two halves as two layers, stable first, and only the
  development stage copies the development part; no RUN uses a BuildKit cache mount
  (Railway).

CI: the image paths of test.yml and images.yml name the input, the locks and the
splitter; images.yml compares the pip layers with ``edge`` on pushes to main only; start.sh
installs the lock.

File reads, plus the two RUNs under ``sh`` in tmp_path with fake tools; no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import json
import marshal
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
INSTALL = ('uv pip install --python /usr/local/bin/python --prefix "/install/$part" --no-deps --require-hashes '
           '--no-compile-bytecode --no-cache --only-binary :all: --no-binary pysher -r "$part.txt"')
PARTS = ("stable", "rest", "development")
COMPILE = ('PYTHONHASHSEED=0 python -m compileall -q -s "$prefix" -p /usr/local --invalidation-mode unchecked-hash '
           '"$prefix/lib/python3.11/site-packages"')
UNCHECKED_HASH_FLAGS = 0b01  # PEP 552: hash-based, source never checked
NEEDS_SH = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the Docker build has")
# What the API's install reads: the image paths in both workflows must name each one.
LOCK_INPUTS = ("orchestrator/requirements.in", "orchestrator/requirements.lock",
               "orchestrator/requirements-dev.txt", "orchestrator/requirements-dev.lock",
               "orchestrator/scripts/split_lock.py")
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_UV_STAGE = re.compile(r"^FROM ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+@sha256:[0-9a-f]{64} AS uv$")

_split_spec = importlib.util.spec_from_file_location("split_lock", ORCH / "scripts" / "split_lock.py")
split_lock = importlib.util.module_from_spec(_split_spec)
_split_spec.loader.exec_module(split_lock)


def _instructions(text: str = DOCKERFILE) -> list[str]:
    """Instructions with continuation lines joined, comment lines dropped, spaces collapsed."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _stage(name: str) -> list[str]:
    instructions = _instructions()
    start = next(i for i, line in enumerate(instructions) if re.match(rf"FROM\s.*\sas\s+{name}$", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start:end]


def _run(marker: str) -> str:
    runs = [line for line in _stage("pybuild") if line.startswith("RUN ") and marker in line]
    assert len(runs) == 1, f"one pybuild RUN holds {marker!r}: {runs}"
    return runs[0]


# --------------------------------------------------------------------------- the Dockerfile


def test_uv_comes_from_a_pinned_named_stage_into_pybuild_only():
    instructions = _instructions()
    stages = [line for line in instructions if line.upper().startswith("FROM ")]
    assert _UV_STAGE.match(stages[0]), stages[0]
    copies = [line for line in instructions if "--from=uv" in line]
    assert copies == ["COPY --from=uv /uv /usr/local/bin/uv"]
    assert copies[0] in _stage("pybuild")
    assert not [line for line in instructions if "pip install uv" in line]


def test_every_build_installs_the_one_lock():
    """PRD-262 US-027: Leiden through graspologic-native; no build arg picks a [leiden] lock."""
    pybuild = _stage("pybuild")
    assert not [line for line in pybuild if line.startswith("ARG ")], pybuild
    assert "COPY requirements.lock scripts/split_lock.py ./" in pybuild
    run = _run("uv pip install")
    assert run.startswith("RUN python split_lock.py requirements.lock stable.txt rest.txt development.txt && "), run
    assert "sed " not in run and "requirements.build" not in DOCKERFILE
    assert "leiden" not in "\n".join(_instructions()).lower()


def test_the_install_is_the_lock_exactly_in_two_halves_and_the_development_part():
    run = _run("uv pip install")
    assert run.endswith(f"for part in {' '.join(PARTS)}; do {INSTALL} || exit 1; done"), run
    assert "--system" not in run, "it would install into pybuild's /usr/local and leave /install empty"
    assert "|| true" not in run
    assert not [line for line in _instructions() if line.startswith("RUN --mount")], "Railway: no cache mounts"


def test_site_packages_compiles_after_the_strip_and_the_prune():
    pybuild = _stage("pybuild")
    at = pybuild.index(_run("compileall"))
    assert at > pybuild.index(_run("strip --strip-unneeded")) and at > pybuild.index(_run("botocore/data"))
    assert at == len(pybuild) - 1, "the last pybuild step: nothing changes site-packages after it"
    prefixes = " ".join(f"/install/{part}" for part in PARTS)
    assert _run("compileall") == f"RUN for prefix in {prefixes}; do {COMPILE} || exit 1; done"


def test_the_base_stage_copies_the_two_halves_stable_first():
    base = _stage("base")
    copies = [line for line in base if line.startswith("COPY ")]
    assert copies == ["COPY --from=pybuild /install/stable /usr/local", "COPY --from=pybuild /install/rest /usr/local"]
    from_pybuild = [line for line in _instructions() if line.startswith("COPY --from=pybuild ")]
    assert [line for line in from_pybuild if line not in copies] == [
        "COPY --from=pybuild /install/development /usr/local"]
    assert from_pybuild[-1] in _stage("development") and from_pybuild[-1] not in _stage("production")


# --------------------------------------------------------------------------- the two RUNs, under sh


def _tools(tmp_path: Path) -> dict:
    """A bin directory with ``python`` (this interpreter) and a ``uv`` that logs its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tools = {"python": f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
             "uv": '#!/bin/sh\necho "$*" >> "$UV_LOG"\ncase "$*" in *"/install/$UV_FAIL "*) exit 1 ;; esac\n'}
    for name, script in tools.items():
        (bin_dir / name).write_text(script, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "UV_LOG": str(tmp_path / "uv.log"), "UV_FAIL": "none"}


def _install(tmp_path: Path, fail: str = "none") -> subprocess.CompletedProcess:
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ORCH / "requirements.lock", work / "requirements.lock")
    shutil.copy(ORCH / "scripts" / "split_lock.py", work / "split_lock.py")
    env = {**_tools(tmp_path), "UV_FAIL": fail}
    script = _run("uv pip install").removeprefix("RUN ")
    return subprocess.run(["sh", "-c", script], cwd=work, env=env, capture_output=True, text=True, timeout=60,
                          check=False)


@NEEDS_SH
def test_the_install_run_installs_the_parts_of_the_lock(tmp_path):
    done = _install(tmp_path)
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "uv.log").read_text().splitlines()
    assert [call.split("--prefix ")[1].split()[0] for call in calls] == [f"/install/{part}" for part in PARTS]
    lock = (ORCH / "requirements.lock").read_text(encoding="utf-8")
    stable, rest = split_lock.split(lock)
    assert (tmp_path / "work" / "stable.txt").read_text() == stable
    assert (tmp_path / "work" / "rest.txt").read_text() == rest
    assert (tmp_path / "work" / "development.txt").read_text() == split_lock.development(lock)
    assert "graspologic-native==" in rest and "graspologic==" not in stable + rest, "US-027"


@NEEDS_SH
def test_a_failed_half_fails_the_install_run(tmp_path):
    done = _install(tmp_path, fail="stable")
    assert done.returncode != 0
    assert len((tmp_path / "uv.log").read_text().splitlines()) == 1, "the second half never starts"


def _compile(tmp_path: Path, files: dict[str, str]) -> subprocess.CompletedProcess:
    for relative, source in files.items():
        path = tmp_path / "install" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    script = _run("compileall").removeprefix("RUN ").replace("/install/", f"{tmp_path}/install/")
    return subprocess.run(["sh", "-c", script], env=_tools(tmp_path), capture_output=True, text=True, timeout=60,
                          check=False)


SITE = "lib/python3.11/site-packages"


@NEEDS_SH
def test_the_compile_run_writes_unchecked_hash_bytecode_named_for_usr_local(tmp_path):
    done = _compile(tmp_path, {f"stable/{SITE}/pkg_a/__init__.py": "A = {'x', 'y'}\n",
                               f"rest/{SITE}/pkg_b/mod.py": "def f():\n    return 1\n"})
    assert done.returncode == 0, done.stdout + done.stderr
    tag = sys.implementation.cache_tag
    for half, module, name in (("stable", "pkg_a", "__init__"), ("rest", "pkg_b", "mod")):
        data = (tmp_path / "install" / half / SITE / module / "__pycache__" / f"{name}.{tag}.pyc").read_bytes()
        assert int.from_bytes(data[4:8], "little") == UNCHECKED_HASH_FLAGS, half
        code = marshal.loads(data[16:])  # bytecode this test just compiled from its own fixture
        assert code.co_filename == f"/usr/local/{SITE}/{module}/{name}.py", code.co_filename


@NEEDS_SH
def test_a_file_that_does_not_compile_fails_the_compile_run(tmp_path):
    done = _compile(tmp_path, {f"stable/{SITE}/pkg_a/__init__.py": "A = 1\n",
                               f"rest/{SITE}/pkg_b/template.py": "def {{ name }}():\n"})
    assert done.returncode != 0


# --------------------------------------------------------------------------- CI


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", LOCK_INPUTS)
def test_the_image_paths_name_the_input_the_locks_and_the_splitter(path):
    changes = next(step for step in _workflow("test.yml")["jobs"]["changes"]["steps"]
                   if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    assert re.search(changes["env"]["PROD_IMAGE_PATHS"], path), "prod-image-parity builds the Railway variant"
    images = _workflow("images.yml")
    assert path in images.get("on", images.get(True))["pull_request"]["paths"]  # PyYAML reads `on` as True
    pick = next(step for step in images["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    api = next(image for image in json.loads(pick["env"]["IMAGES"]) if image["name"] == "api")
    assert path in api["inputs"]


def test_images_yml_compares_the_pip_layers_with_edge_on_main_pushes_only():
    steps = _workflow("images.yml")["jobs"]["build"]["steps"]
    measure = next(i for i, step in enumerate(steps) if step.get("id") == "measure")
    at, step = next((i, step) for i, step in enumerate(steps) if "image_pip_layers.py" in step.get("run", ""))
    assert measure < at
    for condition in ("matrix.name == 'api'", "github.event_name == 'push'", "github.ref == 'refs/heads/main'",
                      "steps.measure.outcome == 'success'"):
        assert condition in step["if"], condition
    assert "schedule" not in step["if"], "the weekly run rebuilds the OS stages and is not compared"
    assert step["env"]["REPORT"] == "${{ runner.temp }}/layers/${{ matrix.name }}-${{ matrix.arch }}.json"
    run = " ".join(_CONTINUATION.sub(" ", step["run"]).split())
    assert "for path in orchestrator/Dockerfile orchestrator/requirements.lock orchestrator/scripts/split_lock.py" in run
    assert "contents/${path}?ref=${revision}" in run
    assert 'image_pip_layers.py --report "$REPORT" --edge-dir "$edge_dir"' in run
    assert "continue-on-error" not in step


def test_start_sh_installs_the_lock():
    start = (ORCH / "start.sh").read_text(encoding="utf-8")
    assert "pip install --require-hashes -r requirements.lock" in start
    assert "requirements.txt" not in start
