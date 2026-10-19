"""PRD-262 US-014: test tools leave the production image.

pytest, pytest-asyncio, pytest-timeout, pytest-cov, black, isort and import-linter sat in
requirements.txt, so every image installed them, and the production stage's
`pip uninstall` RUN removed only four of the seven while adding a layer of its own.
They now live in requirements-dev.txt, which only CI installs. Since PRD-262 US-022 the
API's dependencies are requirements.in, compiled into requirements.lock (and the
Railway [leiden] variant's lock), and the dev file into requirements-dev.lock. This
suite checks:

- the seven tools are in requirements-dev.txt and not in requirements.in or the image's
  locks, and the two inputs name no package twice;
- the Dockerfile copies the image locks alone (never a dev file), has no `pip uninstall`,
  and its production stage compiles the stdlib once, as root, excluding site-packages,
  before the app code is copied in; the only other compile is pybuild's, of
  site-packages;
- every CI install of the API's lock installs the dev lock in the same pip command, hashes
  required, and every pip cache key on the lock lists the dev lock too, each path a file;
- the image workflows watch the dev file and its lock;
- CI checks the built image: `import pytest` fails in it (test.yml's prod-image-parity),
  and the stdlib compile's layer holds stdlib .pyc and nothing under site-packages
  (images.yml, from US-001's layer listing).

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
OSS_SCANNER = (ROOT / ".oss-scanner" / "Dockerfile").read_text(encoding="utf-8")
DEV_FILE = "requirements-dev.txt"
DEV_LOCK = "requirements-dev.lock"
INPUT = "requirements.in"
IMAGE_LOCKS = ("requirements.lock", "requirements-leiden.lock")
TOOLS = {"pytest", "pytest-asyncio", "pytest-timeout", "pytest-cov", "black", "isort", "import-linter"}
STDLIB = "/usr/local/lib/python3.11"
STDLIB_COMPILE = (
    f"RUN python -m compileall -q -x '/site-packages/' --invalidation-mode unchecked-hash {STDLIB}"
)
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
# One pip command that reads one of orchestrator's requirement files (from orchestrator/ or
# the repository root), in a workflow `run` or a RUN; it ends at the line or at && ; |.
_INSTALL = re.compile(r"pip install[^\n&;|]*?-r ((?:orchestrator/)?)requirements[\w.-]*(?=\s|$)[^\n&;|]*", re.M)


def _names(path: Path) -> set[str]:
    """The distribution names a requirements file lists, normalised as PEP 503 does."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _NAME.match(line.split("#", 1)[0].strip())
        if match:
            names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names


def _instructions(text: str) -> list[str]:
    """The Dockerfile's instructions, continuation lines joined and comments dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _stage(text: str, name: str) -> list[str]:
    """The instructions of the stage ``name``, from its FROM to the next."""
    instructions = _instructions(text)
    start = next(i for i, line in enumerate(instructions) if re.match(rf"FROM\s.*\sas\s+{name}$", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start:end]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _steps():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job in (yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}).values():
            for step in job.get("steps") or []:
                yield path.name, step


# --------------------------------------------------------------------------- the two files


def test_the_test_tools_moved_to_the_dev_file():
    assert TOOLS <= _names(ORCH / DEV_FILE), "requirements-dev.txt lacks a test tool"
    assert not TOOLS & _names(ORCH / INPUT), "a test tool is still in requirements.in"
    for lock in IMAGE_LOCKS:
        assert not TOOLS & _names(ORCH / lock), f"the image installs {lock}, which pins a test tool"
    assert TOOLS <= _names(ORCH / DEV_LOCK), "requirements-dev.lock lacks a test tool"


def test_the_two_files_name_no_package_twice():
    """Their locks install in one pip command; a package asked for in both could be pinned two ways."""
    assert not _names(ORCH / DEV_FILE) & _names(ORCH / INPUT)


# --------------------------------------------------------------------------- the image


def test_the_image_never_installs_the_dev_file():
    instructions = "\n".join(_instructions(DOCKERFILE))
    assert DEV_FILE not in instructions and DEV_LOCK not in instructions
    copies = [line for line in _instructions(DOCKERFILE) if re.match(r"COPY\s+requirements", line)]
    assert copies == ["COPY requirements.lock requirements-leiden.lock scripts/split_lock.py ./"], copies


def test_no_pip_uninstall_is_left():
    """The uninstall RUN added a layer instead of removing one, and missed three tools."""
    assert not [line for line in _instructions(DOCKERFILE) if "pip uninstall" in line]


def test_the_production_stage_compiles_the_stdlib_once_as_root_before_the_app():
    production = _stage(DOCKERFILE, "production")
    compiles = [line for line in production if "compileall" in line]
    assert compiles == [STDLIB_COMPILE], "one stdlib compile, site-packages excluded, unchecked-hash"
    pybuild = _stage(DOCKERFILE, "pybuild")
    elsewhere = [line for line in _instructions(DOCKERFILE) if "compileall" in line and line not in production]
    assert len(elsewhere) == 1 and elsewhere[0] in pybuild, (
        "site-packages' bytecode comes only from pybuild (PRD-262 US-022)")
    at = production.index(STDLIB_COMPILE)
    app = next(i for i, line in enumerate(production) if re.fullmatch(r"COPY (--\S+ )*\. \.", line))
    assert at < app, "before the app code, so a code change reuses the layer"
    users = [i for i, line in enumerate(production) if line.startswith("USER ")]
    assert users and at < users[0], "as root: the automatos user can't write to /usr/local"


# --------------------------------------------------------------------------- CI installs


def _ci_installs() -> list[tuple[str, str, str]]:
    """(file, prefix, command) for every pip install of an orchestrator requirement file in CI."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        found += [(path.name, m.group(1), m.group(0)) for m in _INSTALL.finditer(path.read_text(encoding="utf-8"))]
    joined = _CONTINUATION.sub(" ", OSS_SCANNER)
    found += [(".oss-scanner/Dockerfile", m.group(1), m.group(0)) for m in _INSTALL.finditer(joined)]
    return found


def test_every_ci_install_of_the_lock_installs_the_dev_lock_in_the_same_command():
    """PRD-262 US-022: one command, hashes required. A hashed lock puts pip in hash-checking
    mode for the whole command, so the unhashed dev file can't join it, and a dev install of
    its own could move a locked version."""
    installs = _ci_installs()
    where = {name for name, _, _ in installs}
    assert {"test.yml", "nl2sql-eval-scheduled.yml", ".oss-scanner/Dockerfile"} <= where, where
    assert sum(name == "test.yml" for name, _, _ in installs) >= 4
    for name, prefix, command in installs:
        files = re.findall(r"-r (\S+)", command)
        assert files == [f"{prefix}requirements.lock", f"{prefix}{DEV_LOCK}"], f"{name}: {command}"
        assert "--require-hashes" in command.split(), f"{name}: {command}"


def test_the_oss_scanner_needs_no_separate_pytest_timeout():
    assert "pip install pytest-timeout" not in OSS_SCANNER


def test_every_pip_cache_key_on_the_lock_lists_the_dev_lock():
    keyed = []
    for name, step in _steps():
        paths = str((step.get("with") or {}).get("cache-dependency-path", "")).split()
        if "actions/setup-python" not in step.get("uses", ""):
            continue
        # setup-python fails a step whose cache path matches no file (PRD-262 US-022).
        assert all(list(ROOT.glob(path)) for path in paths if path.startswith("orchestrator/")), f"{name}: {paths}"
        if "orchestrator/requirements.lock" in paths:
            keyed.append(name)
            assert f"orchestrator/{DEV_LOCK}" in paths, f"{name}: {paths!r}"
    assert keyed.count("test.yml") == 4 and "nl2sql-eval-scheduled.yml" in keyed, keyed


# --------------------------------------------------------------------------- the image workflows


def test_the_image_workflows_watch_the_dev_file():
    images = _workflow("images.yml")
    triggers = images.get("on", images.get(True))  # PyYAML reads the `on` key as True
    pick = next(step for step in images["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    api = next(image for image in json.loads(pick["env"]["IMAGES"]) if image["name"] == "api")
    for name in (DEV_FILE, DEV_LOCK):
        assert f"orchestrator/{name}" in triggers["pull_request"]["paths"]
        assert f"orchestrator/{name}" in api["inputs"]


def test_ci_finds_no_test_tool_in_the_built_image():
    steps = _workflow("test.yml")["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "import pytest" in step.get("run", ""))
    run = steps[check]["run"]
    assert built < check
    assert re.search(r"if docker run --rm parity-be python -c 'import pytest'; then\n.*exit 1", run)
    assert f"orchestrator/{DEV_FILE}" in run, "every tool the dev file names is looked for"
    assert re.search(
        r'docker run --rm -i --entrypoint python parity-be - "\$\{tools\[@\]\}" < scripts/ci/image_dists_absent\.py',
        run,
    ), "the image's own Python looks each tool up (PRD-262 US-015 shares the script)"


@pytest.mark.parametrize("flag,value", [
    ("--instruction", "compileall"),
    ("--under", f"{STDLIB.lstrip('/')}/"),
    ("--not-under", f"{STDLIB.lstrip('/')}/site-packages/"),
    ("--suffix", ".pyc"),
])
def test_ci_checks_the_stdlib_compiles_layer_in_the_layer_listing(flag, value):
    steps = _workflow("images.yml")["jobs"]["build"]["steps"]
    measure = next(i for i, step in enumerate(steps) if "image_report.py measure" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "image_report.py layer" in step.get("run", ""))
    assert measure < check
    assert steps[check]["if"] == "matrix.name == 'api'"
    assert re.search(rf"{flag} {re.escape(value)}(\s|$)", steps[check]["run"]), steps[check]["run"]
