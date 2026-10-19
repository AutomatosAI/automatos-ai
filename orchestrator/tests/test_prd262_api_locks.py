"""PRD-262 US-022: the API's dependency set is locked, and each lock satisfies its input.

orchestrator/requirements.in (the old requirements.txt), requirements-leiden.in
(Railway's [leiden] build, until US-027) and requirements-dev.txt are compiled with uv
into requirements.lock, requirements-leiden.lock and requirements-dev.lock, every pin
with its hashes. The image, CI, the OSS scanner and start.sh install the locks. This
suite checks:

- scripts/ci/lock_check.py on the committed files: every lock is the compile of its
  input with the PRD's flags, and the locks installed together agree. It reads the files
  only, so an upstream release never turns it red;
- the same check on small fixtures, one per kind of problem it must catch;
- orchestrator/scripts/split_lock.py, which pybuild runs to install the lock in two
  layers: the stable set the PRD names goes first, every other pin second, nothing is
  dropped or reordered, and it refuses a lock it cannot split safely.

File reads and pure functions; no network, no install.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
SPLIT_SCRIPT = ORCH / "scripts" / "split_lock.py"
# The stable set PRD-262 US-022 names, as PEP 503 names.
PRD_STABLE = {"numpy", "scipy", "grpcio", "cryptography", "sqlalchemy", "pillow", "lxml", "fonttools",
              "uvloop", "asyncpg", "pydantic-core"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lock_check = _load("lock_check", ROOT / "scripts" / "ci" / "lock_check.py")
split_lock = _load("split_lock", SPLIT_SCRIPT)


# --------------------------------------------------------------------------- the committed locks


def test_requirements_txt_became_requirements_in():
    assert not (ORCH / "requirements.txt").exists()
    assert (ORCH / "requirements.in").is_file()


def test_every_committed_lock_satisfies_its_input():
    """Fails when an input changed and its lock was not recompiled, or the other way round."""
    assert lock_check.problems(lock_check.read_files()) == []
    assert lock_check.main() == 0


def test_the_locks_are_compiled_with_the_prds_flags():
    commands = {spec.lock: spec.command for spec in lock_check.LOCKS}
    assert set(commands) == {"orchestrator/requirements.lock", "orchestrator/requirements-leiden.lock",
                             "orchestrator/requirements-dev.lock"}
    for command in commands.values():
        flags = command.split()
        assert "--universal" in flags and "--generate-hashes" in flags, command
        assert flags[flags.index("--python-version") + 1] == "3.11", "US-058 moves it to 3.12, not before"
    assert "-c orchestrator/requirements.lock" in commands["orchestrator/requirements-dev.lock"], (
        "dev tools never move a locked version")
    assert "-c orchestrator/requirements.lock" in commands["orchestrator/requirements-leiden.lock"], (
        "Railway's packages are the published image's where the two share one")
    assert "orchestrator/requirements-leiden.in" in commands["orchestrator/requirements-leiden.lock"]


def test_every_pin_of_every_lock_is_exact_and_hashed():
    for spec in lock_check.LOCKS:
        _, pins = lock_check.parse_lock((ROOT / spec.lock).read_text(encoding="utf-8"))
        assert len(pins) > 10, spec.lock
        assert all(pin.operator == "==" and pin.hashes for pin in pins), spec.lock


# --------------------------------------------------------------------------- the check on fixtures

HASH_A, HASH_B, HASH_C = "a" * 64, "b" * 64, "c" * 64


def _spec(lock: str):
    return next(spec for spec in lock_check.LOCKS if spec.lock == lock)


def _lock(lock: str, pins: list[tuple[str, list[str], list[str]]], command: str | None = None) -> str:
    """A uv-style lock: header, then each ``name==version`` with its hashes and ``# via`` lines."""
    lines = [lock_check.HEADER, f"#    {command or _spec(lock).command}"]
    for pin, hashes, via in pins:
        lines.append(f"{pin} \\")
        lines += [f"    --hash=sha256:{value}" + (" \\" if n < len(hashes) - 1 else "") for n, value in enumerate(hashes)]
        lines += [f"    # via {via[0]}"] if len(via) == 1 else ["    # via"] + [f"    #   {source}" for source in via]
    return "\n".join(lines) + "\n"


def _files(**changes: str) -> dict[str, str]:
    """A consistent set of the six files, with ``changes`` (by short name) swapped in."""
    main_in, leiden_in, dev_in = "-r " + lock_check.IN, "-r " + lock_check.LEIDEN_IN, "-r " + lock_check.DEV_IN
    from_main = "-c " + lock_check.LOCK
    files = {
        "in": "# the API\nfastapi==0.110.0\nhttpx>=0.27  # http\n",
        "leiden_in": "graspologic==3.3.0\n",
        "dev_in": "pytest==7.4.3\n",
        "lock": _lock(lock_check.LOCK, [
            ("anyio==4.0.0", [HASH_A], ["httpx"]),
            ("fastapi==0.110.0", [HASH_B], [main_in]),
            ("httpx==0.27.0", [HASH_C, HASH_A], [main_in]),
            ("packaging==24.0", [HASH_A], ["fastapi"]),
        ]),
        "leiden_lock": _lock(lock_check.LEIDEN_LOCK, [
            ("anyio==4.0.0", [HASH_A], [from_main, "httpx"]),
            ("fastapi==0.110.0", [HASH_B], [from_main, main_in]),
            ("graspologic==3.3.0", [HASH_C], [leiden_in]),
            ("httpx==0.27.0", [HASH_C, HASH_A], [from_main, main_in]),
            ("packaging==24.0", [HASH_A], [from_main, "fastapi"]),
        ]),
        "dev_lock": _lock(lock_check.DEV_LOCK, [
            ("packaging==24.0", [HASH_A], [from_main, "pytest"]),
            ("pytest==7.4.3", [HASH_B], [dev_in]),
        ]),
    }
    files.update(changes)
    paths = {"in": lock_check.IN, "leiden_in": lock_check.LEIDEN_IN, "dev_in": lock_check.DEV_IN,
             "lock": lock_check.LOCK, "leiden_lock": lock_check.LEIDEN_LOCK, "dev_lock": lock_check.DEV_LOCK}
    return {paths[key]: text for key, text in files.items()}


def _one_problem(**changes: str) -> str:
    found = lock_check.problems(_files(**changes))
    assert len(found) == 1, found
    return found[0]


def test_a_consistent_fixture_has_no_problem():
    assert lock_check.problems(_files()) == []


def test_a_lock_compiled_with_other_flags_is_caught():
    command = _spec(lock_check.LOCK).command.replace(" --generate-hashes", "")
    text = _files()[lock_check.LOCK].replace(_spec(lock_check.LOCK).command, command)
    assert "the header says" in _one_problem(lock=text)


def test_a_pin_without_a_hash_is_caught():
    text = _lock(lock_check.DEV_LOCK, [("packaging==24.0", [HASH_A], ["-c " + lock_check.LOCK, "pytest"]),
                                       ("pytest==7.4.3", [], ["-r " + lock_check.DEV_IN])])
    assert _one_problem(dev_lock=text) == "orchestrator/requirements-dev.lock: pytest==7.4.3 carries no hash"


def test_a_requirement_the_lock_does_not_pin_is_caught():
    found = lock_check.problems(_files(dev_in="pytest==7.4.3\nblack==23.11.0\n"))
    assert found == ["orchestrator/requirements-dev.lock: orchestrator/requirements-dev.txt asks for "
                     "black==23.11.0, which the lock does not pin"]


def test_a_requirement_bumped_without_recompiling_is_caught():
    found = lock_check.problems(_files(dev_in="pytest==8.0.0\n"))
    assert found == ["orchestrator/requirements-dev.lock: pins pytest at 7.4.3; "
                     "orchestrator/requirements-dev.txt asks for pytest==8.0.0"]


def test_a_removed_requirement_left_in_the_lock_is_caught():
    found = lock_check.problems(_files(leiden_in="# graspologic left\n"))
    assert found == ["orchestrator/requirements-leiden.lock: graspologic==3.3.0 is pinned for "
                     "-r orchestrator/requirements-leiden.in, which no longer asks for it: recompile the lock"]


def test_a_dev_pin_that_moves_a_locked_version_is_caught():
    text = _lock(lock_check.DEV_LOCK, [("packaging==25.0", [HASH_A], ["-c " + lock_check.LOCK, "pytest"]),
                                       ("pytest==7.4.3", [HASH_B], ["-r " + lock_check.DEV_IN])])
    found = lock_check.problems(_files(dev_lock=text))
    assert "orchestrator/requirements-dev.lock: pins packaging at 25.0, orchestrator/requirements.lock at 24.0" in found


def test_a_shared_pin_with_no_hash_in_common_is_caught():
    """pip intersects the hashes of a requirement given twice in one command."""
    text = _lock(lock_check.DEV_LOCK, [("packaging==24.0", [HASH_C], ["-c " + lock_check.LOCK, "pytest"]),
                                       ("pytest==7.4.3", [HASH_B], ["-r " + lock_check.DEV_IN])])
    found = lock_check.problems(_files(dev_lock=text))
    assert "orchestrator/requirements-dev.lock: packaging shares no hash with orchestrator/requirements.lock" in found


def test_the_leiden_lock_must_pin_everything_the_published_lock_pins():
    text = _files()[lock_check.LEIDEN_LOCK]
    block = text[text.index("packaging==24.0"):]
    found = lock_check.problems(_files(leiden_lock=text.replace(block, "")))
    assert "orchestrator/requirements-leiden.lock: lacks packaging, which orchestrator/requirements.lock pins" in found


def test_an_input_option_line_is_refused_rather_than_skipped():
    with pytest.raises(ValueError, match="option line"):
        lock_check.parse_input("-r other.txt\nfastapi==0.110.0\n")


def test_a_lock_line_that_is_not_a_pin_is_refused():
    with pytest.raises(ValueError, match="not a pin"):
        lock_check.parse_lock(f"{lock_check.HEADER}\n#    uv pip compile x\n-e ./local\n")


# --------------------------------------------------------------------------- split_lock.py


def test_the_stable_set_is_the_prds():
    assert set(split_lock.STABLE) == PRD_STABLE


@pytest.mark.parametrize("lock", ["requirements.lock", "requirements-leiden.lock"])
def test_the_split_keeps_every_pin_once_in_order_stable_first(lock):
    text = (ORCH / lock).read_text(encoding="utf-8")
    stable, rest = split_lock.split(text)
    every = split_lock.blocks(text)
    first, second = split_lock.blocks(stable), split_lock.blocks(rest)
    assert {block.name for block in first} == PRD_STABLE
    assert not {block.name for block in second} & PRD_STABLE
    assert first == [block for block in every if block.name in PRD_STABLE]
    assert second == [block for block in every if block.name not in PRD_STABLE]
    assert all("--hash=sha256:" in block.text for block in first + second)
    assert "botocore" in {block.name for block in second}, "US-021's prune works on the second half"


def test_the_split_refuses_a_pin_without_a_hash():
    text = "numpy==2.0.0 \\\n    --hash=sha256:" + HASH_A + "\nrequests==2.0.0\n"
    with pytest.raises(ValueError, match="pins without a hash: requests"):
        split_lock.split(text)


def test_the_split_refuses_a_lock_missing_a_stable_package():
    text = "".join(f"{name}==1.0.0 \\\n    --hash=sha256:{HASH_A}\n" for name in sorted(PRD_STABLE - {"lxml"}))
    with pytest.raises(ValueError, match="does not pin: lxml"):
        split_lock.split(text + f"requests==2.0.0 \\\n    --hash=sha256:{HASH_B}\n")


def test_the_split_refuses_a_lock_with_nothing_outside_the_stable_set():
    text = "".join(f"{name}==1.0.0 \\\n    --hash=sha256:{HASH_A}\n" for name in sorted(PRD_STABLE))
    with pytest.raises(ValueError, match="second layer would be empty"):
        split_lock.split(text)


def test_the_split_refuses_a_line_it_cannot_read():
    with pytest.raises(ValueError, match="line 2"):
        split_lock.blocks(f"numpy==1.0.0 --hash=sha256:{HASH_A}\n--index-url https://example.invalid\n")


def test_the_split_reads_pep_503_names():
    text = f"Pydantic_Core==2.0.0 \\\n    --hash=sha256:{HASH_A}\n"
    assert [block.name for block in split_lock.blocks(text)] == ["pydantic-core"]


def test_the_command_writes_both_halves_and_fails_cleanly(tmp_path, capsys):
    stable, rest = tmp_path / "stable.txt", tmp_path / "rest.txt"
    assert split_lock.main([str(ORCH / "requirements.lock"), str(stable), str(rest)]) == 0
    assert stable.read_text().startswith(("asyncpg==", "cryptography=="))
    assert "botocore==" in rest.read_text()
    broken = tmp_path / "broken.lock"
    broken.write_text("requests==2.0.0\n")
    assert split_lock.main([str(broken), str(stable), str(rest)]) == 1
    assert "pins without a hash" in capsys.readouterr().err
    assert split_lock.main([]) == 2


def test_the_split_runs_on_pybuilds_bare_python():
    """pybuild runs it before anything is installed: the standard library only."""
    tree = ast.parse(SPLIT_SCRIPT.read_text(encoding="utf-8"))
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module and not node.level}
    assert names - {"__future__"} <= set(sys.stdlib_module_names), names
