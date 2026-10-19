"""PRD-262 US-042: a repo check fails on any newly tracked file over 5 MB.

``scripts/ci/check_large_files.py`` compares a change's base with its head and fails
on each file over the limit that is new at its path, or that grew past the limit. A
file already over the limit at the base may change. ``.github/workflows/large-files.yml``
runs it on every pull request and every push to main.

This suite checks:

* the check, on a throwaway git repository with a 100-byte limit: a new large file, a
  renamed one and one that grew fail; a small file, a large file that only changed and
  a deleted one pass; the command's exit codes;
* the lane: every pull request and every push to main, the 5 MB default, the base it
  diffs against;
* this repository's tree: the one file over 5 MB is the launch video US-042 moves to a
  release asset. A second one fails here too, behind any lane that let it in.

git only: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("check_large_files", _REPO / "scripts" / "ci" / "check_large_files.py")
check = importlib.util.module_from_spec(_SPEC)
sys.modules["check_large_files"] = check  # before exec: dataclasses look their module up
_SPEC.loader.exec_module(check)

WORKFLOW_PATH = _REPO / ".github" / "workflows" / "large-files.yml"
WORKFLOW = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
LIMIT = 100  # bytes, for the throwaway repository
SMALL = b"x" * LIMIT
LARGE = b"y" * (LIMIT + 1)

# Tracked files over the 5 MB limit today, each with why it is still here. An entry
# leaves when its file does; a new file over the limit fails this suite.
TRACKED_OVER_THE_LIMIT = {
    "docs/assets/automatos-oss-launch-16x9.mp4": (
        "44.4 MB and unreferenced in the repo. It moves to a release asset once the links "
        "to it outside the repo (the openlaunch kit) point there (PRD-262 US-042)."
    ),
}


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.hooksPath=/dev/null",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"git {args[0]} failed: {proc.stderr}"
    return proc.stdout.strip()


def _commit(repo: Path, files: dict, removed: tuple = ()) -> str:
    for name, data in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in removed:
        (repo / name).unlink()
    _git(repo, "add", "-A", ".")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


@pytest.fixture
def base(repo: Path) -> str:
    return _commit(repo, {"small.txt": SMALL, "old-large.bin": LARGE, "grows.txt": SMALL,
                          "doomed-large.bin": LARGE, "moves-large.bin": LARGE})


def _paths(found) -> dict:
    return {item.path: item.added for item in found}


# --------------------------------------------------------------------------- the check


def test_a_new_large_file_fails_and_a_new_small_one_passes(repo, base):
    head = _commit(repo, {"new-large.mp4": LARGE, "new-small.txt": SMALL, "nested/dir/also-large.png": LARGE})
    assert _paths(check.large_files(base, head, LIMIT, repo)) == {
        "new-large.mp4": True, "nested/dir/also-large.png": True}


def test_a_file_that_grows_past_the_limit_fails(repo, base):
    head = _commit(repo, {"grows.txt": LARGE})
    assert _paths(check.large_files(base, head, LIMIT, repo)) == {"grows.txt": False}


def test_a_file_already_over_the_limit_may_change_or_go(repo, base):
    head = _commit(repo, {"old-large.bin": LARGE + b"more"}, removed=("doomed-large.bin",))
    assert check.large_files(base, head, LIMIT, repo) == []


def test_a_renamed_large_file_counts_as_newly_tracked_at_its_new_path(repo, base):
    _git(repo, "mv", "moves-large.bin", "moved/large.bin")
    head = _commit(repo, {})
    assert _paths(check.large_files(base, head, LIMIT, repo)) == {"moved/large.bin": True}


def test_exactly_the_limit_passes(repo, base):
    head = _commit(repo, {"at-the-limit.bin": b"z" * LIMIT})
    assert check.large_files(base, head, LIMIT, repo) == []


def test_the_report_names_the_file_and_its_size():
    item = check.LargeFile(path="docs/a.mp4", size=44_400_103, added=True)
    assert item.describe(check.LIMIT_BYTES) == "docs/a.mp4 is newly tracked at 44.4 MB (44,400,103 bytes)"
    grown = check.LargeFile(path="a.bin", size=6_000_000, added=False)
    assert grown.describe(check.LIMIT_BYTES) == "a.bin grew past 5 MB at 6.0 MB (6,000,000 bytes)"


def test_the_command_exits_1_on_a_large_file_and_0_without(repo, base, monkeypatch, capsys):
    monkeypatch.setattr(check, "REPO", repo)  # the command runs git in this repository
    small_head = _commit(repo, {"fine.txt": SMALL})
    assert check.main(["--base", base, "--head", small_head, "--limit-bytes", str(LIMIT)]) == 0
    large_head = _commit(repo, {"big.mp4": LARGE})
    assert check.main(["--base", small_head, "--head", large_head, "--limit-bytes", str(LIMIT)]) == 1
    out = capsys.readouterr().out
    assert "::error file=big.mp4::big.mp4 is newly tracked" in out


def test_the_command_exits_2_when_git_cannot_answer(repo, base, monkeypatch):
    monkeypatch.setattr(check, "REPO", repo)
    assert check.main(["--base", "no-such-rev", "--head", base]) == 2


def test_the_limit_is_5_mb():
    assert check.LIMIT_BYTES == 5_000_000


# --------------------------------------------------------------------------- the lane


def test_the_lane_runs_on_every_pull_request_and_every_push_to_main():
    triggers = WORKFLOW[True]  # PyYAML reads the bare key `on` as True
    assert triggers["pull_request"] is None, "no paths or branches filter: every pull request"
    assert triggers["push"] == {"branches": ["main"]}


def test_the_lane_runs_the_check_at_its_default_limit_against_the_base():
    (job,) = WORKFLOW["jobs"].values()
    steps = job["steps"]
    checkout = steps[0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 2, "a pull request's merge commit and its first parent"
    assert checkout["with"]["persist-credentials"] is False
    base_step = next(step for step in steps if step.get("id") == "base")
    assert "HEAD^1" in base_step["run"] and 'fetch --no-tags --depth=1 origin "$BEFORE"' in base_step["run"]
    run = steps[-1]["run"]
    assert run == 'python3 scripts/ci/check_large_files.py --base "$BASE" --head HEAD'
    assert "--limit-bytes" not in WORKFLOW_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- this tree


def test_no_tracked_file_is_over_5_mb_but_the_known_one():
    listing = _git(_REPO, "ls-tree", "-r", "-l", "--full-tree", "HEAD")
    sizes = {}
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        size = meta.split()[3]
        if size != "-":  # a submodule's commit has no size
            sizes[path] = int(size)
    assert "orchestrator/main.py" in sizes, "git ls-tree listed the wrong tree"
    over = {path for path, size in sizes.items() if size > check.LIMIT_BYTES}
    assert over <= set(TRACKED_OVER_THE_LIMIT), (
        f"tracked over {check.LIMIT_BYTES:,} bytes: {sorted(over - set(TRACKED_OVER_THE_LIMIT))}. "
        "Put large media in a release asset, not in the repository."
    )
    assert all(reason.strip() for reason in TRACKED_OVER_THE_LIMIT.values())
