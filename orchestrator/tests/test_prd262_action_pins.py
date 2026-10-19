"""PRD-262 US-012: every action a workflow runs is pinned to a commit.

``scripts/ci/check_action_pins.py`` runs in the action-pins workflow, online. These
tests run it offline on this repository and on fixtures, with a fake GitHub:

* every ``uses:`` in .github/workflows names a 40-hex commit, and gitleaks-action
  is v3.0.0 by SHA (Node 24);
* the line scan finds every ``uses:`` GitHub would run (it agrees with a YAML walk);
* a pinned composite action is read at its commit, and an unpinned ``uses:`` inside
  it fails, followed through nested composites and reusable workflows;
* an exception needs a reason, and one that matches nothing fails;
* a file GitHub would not serve fails the run, never passes it;
* the workflow runs the check on every change to a workflow, the check or its list.

Pure file reads: no network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA = "a" * 40
OTHER_SHA = "b" * 40
GITLEAKS_V3 = "gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e # v3.0.0"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pins = _load("check_action_pins")


def _repo(tmp_path: Path, workflow: str, exceptions: list | None = None) -> Path:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(workflow)
    if exceptions is not None:
        (tmp_path / pins.EXCEPTIONS_FILE).write_text(json.dumps({"exceptions": exceptions}))
    return tmp_path


def _fake_github(files: dict):
    """A fetch that serves ``{(repo, path, commit): text}`` and 404s the rest."""
    asked = []

    def fetch(repo: str, path: str, commit: str):
        asked.append((repo, path, commit))
        return files.get((repo, path, commit))

    fetch.asked = asked
    return fetch


COMPOSITE = "runs:\n  using: 'composite'\n  steps:\n{steps}"
NODE = "runs:\n  using: node24\n  main: dist/index.js\n"


def _composite(*uses: str) -> str:
    return COMPOSITE.format(steps="".join(f"    - uses: {ref}\n" for ref in uses))


# ── this repository ─────────────────────────────────────────────────────────


def test_every_uses_in_this_repository_is_pinned_to_a_commit(capsys):
    assert pins.main(["--offline"]) == 0, capsys.readouterr().out


def test_gitleaks_runs_v3_by_its_commit():
    assert GITLEAKS_V3 in (WORKFLOWS / "gitleaks.yml").read_text()


def _yaml_uses(path: Path) -> list:
    """What GitHub runs: each job's ``uses`` and each step's ``uses``."""
    jobs = (yaml.safe_load(path.read_text()) or {}).get("jobs") or {}
    found = [job["uses"] for job in jobs.values() if "uses" in job]
    found += [step["uses"] for job in jobs.values() for step in job.get("steps") or [] if "uses" in step]
    return found


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.y*ml")), ids=lambda p: p.name)
def test_the_line_scan_finds_every_uses_github_runs(path):
    scanned = [use.ref for use in pins.uses_in(path.read_text(), path.name)]
    assert Counter(scanned) == Counter(_yaml_uses(path))


def test_the_workflow_runs_the_check_on_every_change_it_reads():
    workflow = yaml.safe_load((WORKFLOWS / "action-pins.yml").read_text())
    triggers = workflow[True]  # PyYAML reads the bare key `on` as True
    for event in ("pull_request", "push"):
        paths = set(triggers[event]["paths"])
        assert {
            ".github/workflows/**",
            ".github/actions/**",
            ".github/action-pin-exceptions.json",
            "scripts/ci/check_action_pins.py",
        } <= paths, event
    (step,) = [s for s in workflow["jobs"]["pins"]["steps"] if "run" in s]
    assert step["run"].strip() == "python3 scripts/ci/check_action_pins.py"
    assert "--offline" not in step["run"]
    assert step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"


def test_the_committed_exception_list_reads_and_every_entry_has_a_reason():
    allowed = pins.load_exceptions(ROOT / pins.EXCEPTIONS_FILE)
    assert all(reason.strip() for reason in allowed.values())


# ── the rule ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ref", [
    f"actions/checkout@{SHA}",
    f"github/codeql-action/init@{SHA}",
    f"owner/repo/.github/workflows/build.yml@{SHA}",
    "./.github/actions/local",
    "docker://alpine:3.24@sha256:" + "c" * 64,
])
def test_pinned_references_pass(ref):
    assert pins.pin_problem(ref) is None


@pytest.mark.parametrize("ref", [
    "actions/checkout@v4",
    "actions/checkout@main",
    f"actions/checkout@{SHA[:12]}",
    f"actions/checkout@{SHA.upper()}",
    "actions/checkout",
    f"checkout@{SHA}",
    "docker://alpine:3.24",
])
def test_moving_or_malformed_references_fail(ref):
    assert pins.pin_problem(ref)


def test_an_unpinned_uses_fails_with_its_file_and_line(tmp_path, capsys):
    root = _repo(tmp_path, "jobs:\n  a:\n    steps:\n      - uses: actions/checkout@v4\n")
    assert pins.main(["--root", str(root), "--offline"]) == 1
    assert "::error file=.github/workflows/ci.yml,line=4," in capsys.readouterr().out


def test_quoted_and_commented_references_are_read(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: 'actions/checkout@{SHA}'  # v4.4.0\n")
    assert [use.ref for use in pins.repo_uses(root)] == [f"actions/checkout@{SHA}"]


# ── composites, read at their commit ────────────────────────────────────────


def test_a_node_action_is_read_and_passes(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: actions/checkout@{SHA}\n")
    fetch = _fake_github({("actions/checkout", "action.yml", SHA): NODE})
    assert pins.check(root, fetch, {}) == []
    assert fetch.asked == [("actions/checkout", "action.yml", SHA)]


def test_an_unpinned_uses_inside_a_composite_fails(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: acme/setup@{SHA}\n")
    fetch = _fake_github({("acme/setup", "action.yml", SHA): _composite("actions/cache@v4")})
    (finding,) = pins.check(root, fetch, {})
    assert finding.use.ref == "actions/cache@v4"
    assert finding.use.where.startswith(f"acme/setup/action.yml@{SHA[:12]}")


def test_nested_composites_are_followed_and_action_yaml_is_read(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: acme/tools/install@{SHA}\n")
    fetch = _fake_github({
        ("acme/tools", "install/action.yml", SHA): _composite(f"acme/inner@{OTHER_SHA}", "./local"),
        ("acme/inner", "action.yaml", OTHER_SHA): _composite("actions/setup-node@v4"),
    })
    (finding,) = pins.check(root, fetch, {})
    assert finding.use.ref == "actions/setup-node@v4"
    assert ("acme/inner", "action.yml", OTHER_SHA) in fetch.asked


def test_a_reusable_workflows_own_uses_are_checked(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    uses: acme/ci/.github/workflows/build.yml@{SHA}\n")
    reusable = "jobs:\n  b:\n    steps:\n      - uses: actions/checkout@v4\n"
    fetch = _fake_github({("acme/ci", ".github/workflows/build.yml", SHA): reusable})
    (finding,) = pins.check(root, fetch, {})
    assert finding.use.ref == "actions/checkout@v4"


def test_a_commit_with_no_action_file_fails(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: acme/setup@{SHA}\n")
    (finding,) = pins.check(root, _fake_github({}), {})
    assert "no action.yml or action.yaml" in finding.problem


def test_each_action_is_read_once(tmp_path):
    step = f"      - uses: acme/setup@{SHA}\n"
    root = _repo(tmp_path, "jobs:\n  a:\n    steps:\n" + step * 3)
    fetch = _fake_github({("acme/setup", "action.yml", SHA): NODE})
    assert pins.check(root, fetch, {}) == []
    assert len(fetch.asked) == 1


def test_a_file_github_will_not_serve_fails_the_run(tmp_path, capsys):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: acme/setup@{SHA}\n")

    def refused(repo, path, commit):
        raise urllib.error.HTTPError("https://api.github.com", 403, "rate limited", None, None)

    assert pins.main(["--root", str(root)], fetch=refused) == 1
    assert "could not finish" in capsys.readouterr().out


# ── exceptions ──────────────────────────────────────────────────────────────


def test_an_exception_with_a_reason_excuses_its_reference(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: acme/setup@{SHA}\n")
    fetch = _fake_github({("acme/setup", "action.yml", SHA): _composite("vendor/tool@v1")})
    assert pins.check(root, fetch, {"vendor/tool@v1": "the vendor publishes no tags"}) == []


def test_an_exception_without_a_reason_fails(tmp_path, capsys):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: actions/checkout@{SHA}\n",
                 exceptions=[{"uses": "vendor/tool@v1", "reason": " "}])
    assert pins.main(["--root", str(root), "--offline"]) == 1
    assert "needs 'uses' and a 'reason'" in capsys.readouterr().out


def test_an_exception_that_matches_nothing_fails_online(tmp_path):
    root = _repo(tmp_path, f"jobs:\n  a:\n    steps:\n      - uses: actions/checkout@{SHA}\n")
    fetch = _fake_github({("actions/checkout", "action.yml", SHA): NODE})
    (finding,) = pins.check(root, fetch, {"vendor/tool@v1": "gone"})
    assert finding.problem == "this exception matches no reference"
    # Offline, the composites were not read, so an unmatched entry proves nothing.
    assert pins.check(root, None, {"vendor/tool@v1": "gone"}) == []
