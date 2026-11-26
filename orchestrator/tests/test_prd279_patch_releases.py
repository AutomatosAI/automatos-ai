"""PRD-279 US-304 and §5.5: a patch to an older release is cut on ``release/MAJOR.MINOR``.

images.yml's preflight (``scripts/ci/release_preflight.py``) runs on every ``v*`` tag before
anything builds, and lets a version be cut only from a commit on a release line: the default
branch, or ``release/MAJOR.MINOR`` for the tag's own MAJOR.MINOR. Each case, through the
script's own entry point with GitHub's compare answer faked (``ahead`` or ``identical``: the
branch holds the commit; ``behind`` or ``diverged``: it does not; None: no such branch):

* a tag on main builds;
* a patch tag, or a patch's release candidate, on its own release branch builds, though main
  has moved past it;
* a tag on a commit that is on neither stops the run;
* a tag on a release branch of another MAJOR.MINOR stops the run: that branch is never asked;
* a tag whose commit already holds a higher full release (``v0.11.3`` on a main commit after
  ``v0.12.0``) stops the run, naming that release and the release branch to cut from; the
  tags reachable from the commit come from git (faked), and a shallow or unreadable history
  stops the run too (#1658).

And the rest of the tag path treats a patch as any release: its images move their own
``MAJOR.MINOR`` line (a candidate's do not), and nothing on the tag path names main.

Fakes stand in for GitHub and the registry: no network.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
PREFIX = "ghcr.io/automatosai/automatos-"
DOCS = ROOT / "docs" / "deployment-infrastructure"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("release_preflight")
tags = _load("release_tags")


class Compare:
    """GitHub's compare of each branch against the tagged commit, recording what was asked."""

    def __init__(self, statuses: dict):
        self.statuses, self.asked = statuses, []

    def __call__(self, branch: str):
        self.asked.append(branch)
        return self.statuses.get(branch)


class Git:
    """git in the full-history checkout: the version tags reachable from the tagged commit."""

    def __init__(self, reachable=(), shallow=False):
        self.reachable, self.shallow, self.calls = list(reachable), shallow, []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if args == ["rev-parse", "--is-shallow-repository"]:
            return "true\n" if self.shallow else "false\n"
        if args == ["tag", "--list", "--merged", SHA, "v*"]:
            return "".join(f"{tag}\n" for tag in self.reachable)
        raise AssertionError(args)


def _run(version: str, statuses: dict, git=None) -> tuple:
    compare = Compare(statuses)
    args = ["--version", version, "--sha", SHA, "--repo", "AutomatosAI/automatos-ai", "--default-branch", "main",
            "--prefix", PREFIX, "--image", "api", "--image", "frontend"]
    return preflight.main(args, state=lambda ref: False, compare=compare, git=git or Git()), compare.asked


@pytest.mark.parametrize("version", ["0.12.0", "0.12.0-rc.1", "0.11.3"])
def test_a_tag_on_main_builds_and_no_release_branch_is_asked(version, capsys):
    assert _run(version, {"main": "ahead"}) == (0, ["main"])
    assert "::error" not in capsys.readouterr().out


@pytest.mark.parametrize("version,branch", [
    ("0.11.3", "release/0.11"),
    ("0.11.3-rc.1", "release/0.11"),
    ("1.4.12", "release/1.4"),
])
@pytest.mark.parametrize("status", ["ahead", "identical"])
def test_a_patch_tag_on_its_release_branch_builds_though_main_has_moved_on(version, branch, status, capsys):
    code, asked = _run(version, {"main": "diverged", branch: status})
    assert code == 0, capsys.readouterr().out
    assert asked == ["main", branch]


@pytest.mark.parametrize("statuses", [
    {"main": "diverged", "release/0.11": "diverged"},
    {"main": "behind", "release/0.11": "behind"},
    {"main": "diverged"},
    {},
])
def test_a_tag_on_a_commit_on_neither_stops_the_run(statuses, capsys):
    code, _ = _run("0.11.3", statuses)
    assert code == 1
    assert f"{SHA[:12]} is on neither main nor release/0.11" in capsys.readouterr().out


@pytest.mark.parametrize("version,other", [
    ("0.12.1", "release/0.11"),
    ("0.11.3", "release/0.12"),
    ("1.11.3", "release/0.11"),
    ("0.1.3", "release/0.11"),
])
def test_a_tag_on_another_lines_release_branch_stops_the_run(version, other, capsys):
    code, asked = _run(version, {"main": "diverged", other: "ahead"})
    assert code == 1
    assert other not in asked, "only the tag's own MAJOR.MINOR branch is asked"
    assert "a version is cut from main, or from the release branch of its own MAJOR.MINOR" in capsys.readouterr().out


@pytest.mark.parametrize("version,reachable,higher", [
    ("0.11.3", ["v0.11.2", "v0.12.0"], "v0.12.0"),
    ("0.11.3-rc.1", ["v0.11.2", "v0.12.0-rc.1", "v0.12.0", "v0.12.1"], "v0.12.1"),
    ("0.9.0", ["v0.10.0", "v1.0.0", "v0.11.2"], "v1.0.0"),
    ("1.4.12", ["v1.5.0"], "v1.5.0"),
])
def test_a_tag_below_a_release_the_commit_already_holds_stops_the_run(version, reachable, higher, capsys):
    code, _ = _run(version, {"main": "ahead"}, Git(reachable))
    assert code == 1
    line = ".".join(version.split(".")[:2])
    assert f"{SHA[:12]} already holds {higher}, a later release than {version}: cut {version} from release/{line}" \
        in capsys.readouterr().out


@pytest.mark.parametrize("version,reachable", [
    ("0.12.0", ["v0.11.2", "v0.12.0-rc.1"]),
    ("0.12.0", ["v0.11.2", "v0.12.0-rc.1", "v0.12.0"]),
    ("0.12.1", ["v0.12.0", "v0.13.0-rc.1"]),
    ("0.11.3", ["v0.11.2", "v0.11.3-rc.1", "v0.11.3", "not-a-version", "v0.12"]),
    ("0.11.3", []),
])
def test_a_tag_at_or_above_every_release_the_commit_holds_builds(version, reachable, capsys):
    """Its own tag, candidates and anything that is not a full release never count against it."""
    code, _ = _run(version, {"main": "ahead"}, Git(reachable))
    assert code == 0, capsys.readouterr().out


def test_a_patch_on_its_release_branch_builds_beside_a_higher_release_on_main(capsys):
    """release/0.11 is made from v0.11.2: main's v0.12.0 is not reachable from its commits."""
    code, _ = _run("0.11.3", {"main": "diverged", "release/0.11": "ahead"}, Git(["v0.11.0", "v0.11.2"]))
    assert code == 0, capsys.readouterr().out


def test_a_history_that_cannot_be_read_stops_the_run(capsys):
    assert _run("0.11.3", {"main": "ahead"}, Git(shallow=True))[0] == 1
    assert f"could not read the releases reachable from {SHA[:12]}" in capsys.readouterr().out

    def broken(args):
        raise subprocess.CalledProcessError(128, ["git", *args])
    assert _run("0.11.3", {"main": "ahead"}, broken)[0] == 1
    assert f"could not read the releases reachable from {SHA[:12]}" in capsys.readouterr().out


def test_a_patch_moves_its_own_line_and_a_patch_candidate_moves_none():
    digest = "sha256:" + "a" * 64
    plans = tags.plan({"api": digest}, PREFIX, "0.11.3", lambda ref: None)
    assert plans[0].tags == (f"{PREFIX}api:0.11.3", f"{PREFIX}api:0.11")
    assert tags.plan({"api": digest}, PREFIX, "0.11.3-rc.1", lambda ref: None)[0].tags == (f"{PREFIX}api:0.11.3-rc.1",)


def test_the_tag_path_runs_on_any_v_tag_whatever_branch_it_was_cut_on():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["push"]["tags"] == ["v*"]
    assert triggers["push"]["branches"] == ["main"], "a merge into a release branch publishes nothing"
    assert triggers["pull_request"]["branches"] == ["main", "release/**"], "a cherry-pick is built and scanned"
    for job in ("release-images", "release-chart", "release-notes"):
        assert workflow["jobs"][job]["if"] == "startsWith(github.ref, 'refs/tags/v')", job


def test_every_lane_a_pull_request_into_main_runs_runs_into_a_release_branch_too():
    """A patch's pull request runs main's checks (#1658): a lane filtered to main lists release/** too."""
    filtered = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as True
        triggers = triggers if isinstance(triggers, dict) else {}
        assert "pull_request_target" not in triggers, f"{path.name}: a pull request never gets the base's secrets"
        branches = (triggers.get("pull_request") or {}).get("branches")
        if branches is not None:
            filtered.append(path.name)
            assert "release/**" in branches, f"{path.name} runs on pull requests into {branches} only"
    for lane in ("smoke-images.yml", "codeql.yml", "smoke-fresh-clone.yml", "voice-tts.yml"):
        assert lane in filtered, lane


def test_the_patch_page_records_the_branch_the_cherry_pick_the_tag_and_the_branch_setting():
    page = (DOCS / "patching-a-release.md").read_text(encoding="utf-8")
    for part in ("release/MAJOR.MINOR", "git cherry-pick -x", "pull request", "git tag -a v0.11.3-rc.1",
                 "Target branches", "`release/*`", "Require a pull request before merging",
                 "Require status checks to pass", "Block force pushes", "Restrict deletions", "Restrict creations",
                 "A patch changes no database"):
        assert part in page, part
    assert "patching-a-release.md" in (DOCS / "releases.md").read_text(encoding="utf-8")
