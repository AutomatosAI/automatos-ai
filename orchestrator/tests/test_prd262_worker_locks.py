"""PRD-262 US-044: the worker's dependencies are locked, with the SDK in its own layer.

services/workspace-worker/requirements.in lists the worker's direct dependencies.
requirements.lock is their whole resolution: uv compiles it for every platform
(--universal, Python 3.12, because it installs on linux/amd64 and linux/arm64)
with every file's sha256. requirements-browser.lock locks Playwright (US-043)
the same way, against the main lock, so the packages both carry keep one pin.

The Dockerfile installs the locks with the uv binary it copies from the pinned uv
image, with --require-hashes and --no-deps, in three layers: everything except
claude-agent-sdk and psycopg2, then Playwright in the browser step, then the SDK
alone, last. psycopg2 is built from its hash-checked sdist in a stage of its own,
against Debian's libpq, into /install, which the production stage copies (PRD-262,
row 27 of the 19 Oct size breakdown). Its `locks` stage cuts requirements.lock into
those three parts, and this suite runs that awk on the real lock. SQLAlchemy has left the worker
(test_prd262_worker_task_executions.py covers its replacement).

test.yml's prod-image-parity job checks the built image: the SDK version against
the lock, uv against its pin, no SQLAlchemy, and a consistent installed set.

File reads plus awk on the real lock: no Docker, no network.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
REL = "services/workspace-worker"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
MAIN_LOCK = "requirements.lock"
BROWSER_LOCK = "requirements-browser.lock"
SDK = "claude-agent-sdk"
PIN_LINE = re.compile(r"^([a-z0-9][a-z0-9._-]*)==([^\s;\\]+)(?: ; [^\\]+)? \\$")
HASH_LINE = re.compile(r"^    --hash=sha256:[0-9a-f]{64}(?: \\)?$")
UV_STAGE = re.compile(r"^FROM ghcr\.io/astral-sh/uv:(\d+\.\d+\.\d+)@sha256:[0-9a-f]{64} AS uv$", re.M)
UV_INSTALL = re.compile(r"\buv pip install ([^;&]*?)-r (\S+\.lock)")
INSTALL_FLAGS = ("--no-cache", "--compile-bytecode", "--require-hashes", "--no-deps")
# Where each install goes: the image's own Python, or psycopg2's build stage's prefix.
PSYCOPG2_LOCK = "/locks/psycopg2.lock"
TARGET_FLAGS = {PSYCOPG2_LOCK: ("--prefix", "/install")}
SYSTEM = ("--system",)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


INSTRUCTIONS = _instructions(DOCKERFILE)


def _entries(text: str) -> list[list[str]]:
    """The lock's entries: each pin line with the indented lines under it."""
    entries: list[list[str]] = []
    for line in text.splitlines():
        if re.match(r"^[A-Za-z0-9]", line):
            entries.append([line])
        elif entries:
            entries[-1].append(line)
    return entries


def _pins(text: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for entry in _entries(text) if (match := PIN_LINE.match(entry[0]))}


def _lock(name: str) -> str:
    return (WORKER / name).read_text()


def _requirements(name: str) -> list[Requirement]:
    lines = [line.split("#", 1)[0].strip() for line in (WORKER / name).read_text().splitlines()]
    return [Requirement(line) for line in lines if line]


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _index(predicate) -> int:
    found = [i for i, line in enumerate(INSTRUCTIONS) if predicate(line)]
    assert len(found) == 1, f"expected one matching instruction, found {[INSTRUCTIONS[i] for i in found]}"
    return found[0]


# ── the locks ───────────────────────────────────────────────────────────────


def test_the_requirements_are_an_input_and_a_hash_lock():
    assert not (WORKER / "requirements.txt").exists(), "requirements.txt became requirements.in (US-044)"
    command = _lock(MAIN_LOCK).splitlines()[1]
    assert f"uv pip compile {REL}/requirements.in " in command
    for flag in ("--universal", "--python-version 3.12", "--generate-hashes", f"-o {REL}/{MAIN_LOCK}"):
        assert flag in command, flag


def test_the_browser_lock_is_compiled_against_the_main_lock():
    command = _lock(BROWSER_LOCK).splitlines()[1]
    assert f"uv pip compile {REL}/requirements-browser.in " in command
    for flag in (f"-c {REL}/{MAIN_LOCK}", "--universal", "--python-version 3.12", "--generate-hashes",
                 f"-o {REL}/{BROWSER_LOCK}"):
        assert flag in command, flag
    main, browser = _pins(_lock(MAIN_LOCK)), _pins(_lock(BROWSER_LOCK))
    shared = set(main) & set(browser)
    assert "typing-extensions" in shared
    assert {name: (main[name], browser[name]) for name in shared if main[name] != browser[name]} == {}


@pytest.mark.parametrize("lock", [MAIN_LOCK, BROWSER_LOCK])
def test_every_line_of_the_lock_is_an_exact_pin_or_its_hash(lock):
    """Every entry is name==version with at least one sha256; every other line a comment."""
    text = _lock(lock)
    stray = [line for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")
             and not PIN_LINE.match(line) and not HASH_LINE.match(line)]
    assert stray == [], f"lines that are neither an exact pin nor a hash: {stray[:5]}"
    entries = _entries(text)
    assert entries, "the lock pins nothing"
    unhashed = [entry[0] for entry in entries if not any(HASH_LINE.match(line) for line in entry[1:])]
    assert unhashed == [], f"pins without a hash: {unhashed}"
    assert len(_pins(text)) == len(entries)


@pytest.mark.parametrize(("source", "lock"), [("requirements.in", MAIN_LOCK),
                                              ("requirements-browser.in", BROWSER_LOCK)])
def test_each_lock_satisfies_its_input(source, lock):
    pins = _pins(_lock(lock))
    for requirement in _requirements(source):
        name = _canonical(requirement.name)
        assert name in pins, f"{lock} does not pin {name}, which {source} requires"
        assert requirement.specifier.contains(pins[name], prereleases=True), (
            f"{lock} pins {name}=={pins[name]}, outside {source}'s {requirement.specifier}: recompile the lock"
        )


def test_the_sdk_is_pinned_exactly_and_the_lock_agrees():
    (sdk,) = [requirement for requirement in _requirements("requirements.in") if _canonical(requirement.name) == SDK]
    (spec,) = list(sdk.specifier)
    assert spec.operator == "==", f"{SDK} floats ({sdk.specifier}): pin it exactly"
    assert _pins(_lock(MAIN_LOCK))[SDK] == spec.version


def test_sqlalchemy_is_gone_and_greenlet_stays_only_for_playwright():
    main, browser = _pins(_lock(MAIN_LOCK)), _pins(_lock(BROWSER_LOCK))
    assert "sqlalchemy" not in main and "sqlalchemy" not in browser
    assert not [r for r in _requirements("requirements.in") if _canonical(r.name) == "sqlalchemy"]
    assert "greenlet" not in main, "greenlet came with SQLAlchemy; only Playwright needs it now"
    assert "greenlet" in browser and "playwright" in browser
    assert "playwright" not in main


# ── the Dockerfile ──────────────────────────────────────────────────────────


def test_uv_comes_from_a_pinned_stage_to_the_path_agents_use():
    assert UV_STAGE.search(DOCKERFILE), "uv's FROM line must be ghcr.io/astral-sh/uv:<x.y.z>@sha256:<digest> AS uv"
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in INSTRUCTIONS
    unmanaged = [line for line in INSTRUCTIONS if re.search(r"(?<!uv )\bpip3? install\b", line)]
    assert unmanaged == [], f"installs that bypass uv and the locks: {unmanaged}"
    assert "requirements.txt" not in DOCKERFILE


def test_every_install_is_a_hash_checked_lock_with_no_resolution():
    installs = [(line, match) for line in INSTRUCTIONS for match in UV_INSTALL.finditer(line)]
    assert [match.group(2) for _, match in installs] == [
        PSYCOPG2_LOCK, "/opt/worker-locks/stable.lock", BROWSER_LOCK, "/opt/worker-locks/sdk.lock"]
    for _, match in installs:
        words = match.group(1).split()
        missing = [flag for flag in INSTALL_FLAGS if flag not in words]
        assert missing == [], f"{match.group(0)} lacks {missing}"
        target = TARGET_FLAGS.get(match.group(2), SYSTEM)
        assert " ".join(target) in " ".join(words), f"{match.group(0)} does not install into {target}"


def test_the_sdk_installs_last_in_its_own_layer_after_the_browser():
    stable_at = _index(lambda line: line.startswith("RUN uv pip install") and "stable.lock" in line)
    user_at = _index(lambda line: line.startswith("RUN") and "useradd" in line)
    browser_at = _index(lambda line: line.startswith("RUN") and "playwright install" in line)
    sdk_at = _index(lambda line: line.startswith("RUN uv pip install") and "sdk.lock" in line)
    code_at = _index(lambda line: line.startswith("COPY") and line.endswith(" . ."))
    assert stable_at < user_at < browser_at < sdk_at < code_at
    assert INSTRUCTIONS[sdk_at - 1] == "COPY --from=locks /locks/sdk.lock /opt/worker-locks/"
    assert INSTRUCTIONS[stable_at - 1] == "COPY --from=locks /locks/stable.lock /opt/worker-locks/"
    assert "requirements-browser.lock" in INSTRUCTIONS[browser_at]
    later = [line for line in INSTRUCTIONS[sdk_at + 1:] if "install" in line]
    assert later == [], f"an install after the SDK's moves its layer on every bump: {later}"


def _locks_stage_awk() -> dict[str, str]:
    """The `locks` stage's awk programs, by the file each writes."""
    stage = DOCKERFILE.split(" AS locks\n", 1)[1].split("\nFROM ", 1)[0]
    run = re.sub(r"\\[ \t]*\n", " ", stage)
    programs = dict((out, program) for program, out in re.findall(r"awk '([^']+)' requirements\.lock > (\S+)", run))
    assert set(programs) == {"stable.lock", "sdk.lock", "psycopg2.lock"}, programs
    return programs


def test_the_locks_stage_cuts_the_lock_into_the_sdk_psycopg2_and_the_rest():
    awk = shutil.which("awk")
    assert awk, "awk is needed to run the Dockerfile's split"
    lock = _lock(MAIN_LOCK)
    parts = {
        out: subprocess.run([awk, program], input=lock, capture_output=True, text=True, check=True, timeout=30).stdout
        for out, program in _locks_stage_awk().items()
    }
    sdk, stable, psycopg2 = (_pins(parts[name]) for name in ("sdk.lock", "stable.lock", "psycopg2.lock"))
    assert sdk == {SDK: _pins(lock)[SDK]}
    assert psycopg2 == {"psycopg2": _pins(lock)["psycopg2"]}
    assert stable == {name: version for name, version in _pins(lock).items() if name not in {SDK, "psycopg2"}}
    # Nothing lost and nothing doubled: every line of the lock lands in exactly one part.
    assert sum((Counter(part.splitlines()) for part in parts.values()), Counter()) == Counter(lock.splitlines())
    for name in ("sdk.lock", "psycopg2.lock"):
        assert [line for line in parts[name].splitlines() if HASH_LINE.match(line)], f"{name} lost its hashes"


def test_no_cache_mount_in_the_worker_until_railway_gives_its_service_id():
    """PRD-262 §10: the worker's Railway service id is unknown, so no install has a cache mount."""
    assert not [line for line in INSTRUCTIONS if "--mount=type=cache" in line]


def test_the_parity_job_checks_the_installed_set():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())["jobs"]["prod-image-parity"]["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    for needle in (
        "for image in parity-wk parity-wk-local; do",
        'docker run --rm --entrypoint pip "$image" show claude-agent-sdk',
        f"{REL}/{MAIN_LOCK}",
        'docker run --rm --entrypoint uv "$image" --version',
        "ghcr\\.io/astral-sh/uv:",
        "docker run --rm --entrypoint python \"$image\" -c 'import sqlalchemy'",
        'docker run --rm --entrypoint pip "$image" check',
        "_bundled/claude",
    ):
        assert needle in script, needle
