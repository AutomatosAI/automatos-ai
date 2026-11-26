"""PRD-279 US-301: what stops a release page, and that it stops the run before anything is built.

* **A migration is never dropped.** A file in the versions folder that sets ``revision`` or
  ``down_revision`` in a form the reader cannot take as a literal stops the page with the
  file's name; a file that sets neither is skipped.
* **The early check.** On a ``v*`` tag the ``changes`` job runs ``release_page.py check`` before
  any image is built, from a full checkout of the tagged commit, with a read-only token. It
  refuses everything ``publish`` refuses that needs no digest: a shallow checkout, a tag the
  checkout lacks, a missing or malformed upgrade file, ``none`` once there is a release, an
  upgrade version with no tag, an unreadable migration, and a page that exists or an answer
  that is not "no page". It only reads: it never creates a page.
* **The create call** names the tagged commit as its target; a summary that cannot be written
  after the page is created is a warning, not a failed run.
* **Annotations** are escaped, so a path holding a line break or ``::`` cannot start a command.
* **A commit with no message** is listed, by its commit.

Fakes stand in for git and GitHub: no network, and no git history of its own.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = "AutomatosAI/automatos-ai"
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
DIGEST = "sha256:" + "a" * 64
NAMES = ["api", "frontend", "enterprise-frontend", "workspace-worker"]
VERSIONS = "orchestrator/alembic/versions"
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text(encoding="utf-8"))
JOBS = WORKFLOW["jobs"]
ON_A_TAG = "startsWith(github.ref, 'refs/tags/v')"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


history = _load("release_history")
notes_md = _load("release_notes")
page = _load("release_page")


def _source(revision: str, down) -> str:
    return f"revision = {revision!r}\ndown_revision = {down!r}\n"


OLD_TREE = {"r1": _source("r1", None), "r2": _source("r2", "r1")}
NEW_TREE = {**OLD_TREE, "r3": _source("r3", "r2")}
LOG = f"{'c' * 40}\n{'1' * 40}\nfeat(api): a change (#40)\n\0"


class FakeGit:
    """v0.11.0 holds r1 and r2; the tagged commit adds r3. Each tree is ``{stem: source}``."""

    def __init__(self, tags=("v0.11.0", "v0.12.0"), shallow=False, new=None, log=LOG):
        self.tags, self.shallow, self.log, self.calls = list(tags), shallow, log, []
        self.trees = {"refs/tags/v0.11.0": OLD_TREE, SHA: NEW_TREE if new is None else new}

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if args == ["rev-parse", "--is-shallow-repository"]:
            return "true\n" if self.shallow else "false\n"
        if args[:2] == ["tag", "--list"]:
            return "\n".join(self.tags) + "\n"
        if args[0] == "ls-tree":
            return "".join(f"100644 blob {'0' * 40}\t{VERSIONS}/{stem}.py\0" for stem in self.trees[args[2]])
        if args[0] == "show":
            ref, path = args[1].split(":", 1)
            return self.trees[ref][path[len(VERSIONS) + 1:-3]]
        if args[0] == "log":
            return self.log
        raise AssertionError(args)


class FakeGitHub:
    def __init__(self, existing: int = 404):
        self.existing, self.calls = existing, []

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if method == "GET":
            return self.existing, None
        return 201, {"html_url": f"https://github.com/{REPO}/releases/tag/v0.12.0"}


def _argv(tmp_path: Path, command: str, upgrade="0.11.0\n", summary=None) -> list:
    argv = [command, "--repo", REPO, "--version", "0.12.0", "--sha", SHA,
            "--upgrade-file", str(tmp_path / "upgrade.txt")] + [arg for name in NAMES for arg in ("--image", name)]
    if upgrade is not None:
        (tmp_path / "upgrade.txt").write_text(upgrade, encoding="utf-8")
    if summary is not None:
        argv += ["--summary", str(summary)]
    if command == "publish":
        digests = tmp_path / "digests"
        digests.mkdir(exist_ok=True)
        for name in NAMES:
            (digests / name).write_text(DIGEST + "\n", encoding="utf-8")
        argv += ["--digests", str(digests), "--chart-digest", DIGEST, "--notes", str(tmp_path / "notes.md")]
    return argv


# --------------------------------------------------------------------------- a migration is never dropped


FILE = f"{VERSIONS}/odd.py"


@pytest.mark.parametrize("source,message", [
    ("revision = 'r' + '3'\ndown_revision = 'r2'\n", "revision is not a non-empty string literal"),
    ("revision = f'r{3}'\ndown_revision = 'r2'\n", "revision is not a non-empty string literal"),
    ("NAME = 'r3'\nrevision = NAME\ndown_revision = 'r2'\n", "revision is not a non-empty string literal"),
    ("revision = ''\ndown_revision = 'r2'\n", "revision is not a non-empty string literal"),
    ("revision = 3\ndown_revision = 'r2'\n", "revision is not a non-empty string literal"),
    ("if True:\n    revision = 'r3'\ndown_revision = 'r2'\n", "in a form this check cannot read"),
    ("try:\n    revision = 'r3'\nexcept ImportError:\n    pass\ndown_revision = 'r2'\n", "in a form this check cannot read"),
    ("revision, down_revision = 'r3', 'r2'\n", "in a form this check cannot read"),
    ("revision = down_revision = 'r3'\n", "in a form this check cannot read"),
    ("revision = 'r'\nrevision += '3'\ndown_revision = 'r2'\n", "in a form this check cannot read"),
    ("from elsewhere import revision\ndown_revision = 'r2'\n", "in a form this check cannot read"),
    ("revision: str\ndown_revision = 'r2'\n", "in a form this check cannot read"),
    ("revision = 'r3'\ndown_revision = 'r' + '2'\n", "down_revision is not a revision id"),
    ("revision = 'r3'\ndown_revision = ('r2', 1)\n", "down_revision is not a revision id"),
    ("revision = 'r3'\ndown_revision = ''\n", "down_revision is not a revision id"),
    ("revision = 'r3'\ndown_revision = PARENT\n", "down_revision is not a revision id"),
    ("revision = 'r3'\n", "sets revision but not down_revision"),
    ("down_revision = 'r2'\n", "sets down_revision but not revision"),
])
def test_a_migration_that_cannot_be_read_stops_the_page_with_its_name(source, message):
    with pytest.raises(history.HistoryError, match=re.escape(message)) as raised:
        history.revisions({FILE: source})
    assert FILE in str(raised.value)


@pytest.mark.parametrize("source", [
    "HELPER = 1\n",
    "revision_count = 3\n",
    "def upgrade():\n    revision = 'local'\n    return revision\n",
    "class Thing:\n    revision = 'an attribute'\n",
])
def test_a_file_that_sets_neither_name_is_not_a_migration_and_is_skipped(source):
    assert history.revisions({FILE: source}) == {}


@pytest.mark.parametrize("source,parents", [
    ("revision = 'm'\ndown_revision = ('a', 'b')\n", ("a", "b")),
    ("revision = 'm'\ndown_revision = [\n    'a',\n    'b',\n]\n", ("a", "b")),
    ("revision: str = 'm'\ndown_revision: str | None = None\n", ()),
    ('"""Doc.\n\nrevision = "not this"\n"""\nrevision = "m"\ndown_revision = "a"\n', ("a",)),
])
def test_every_plain_form_is_still_read(source, parents):
    assert history.revisions({FILE: source})["m"].parents == parents


def test_a_new_migration_the_reader_cannot_read_never_becomes_no_change(tmp_path, capsys):
    new = {**OLD_TREE, "r3": "revision = 'r' + '3'\ndown_revision = 'r2'\n"}
    assert page.main(_argv(tmp_path, "check"), git=FakeGit(new=new), call=FakeGitHub()) == 1
    out = capsys.readouterr().out
    assert "r3.py: revision is not a non-empty string literal" in out and "does not change" not in out


# --------------------------------------------------------------------------- the early check


def test_the_early_check_passes_and_only_reads(tmp_path, capsys):
    github, summary = FakeGitHub(), tmp_path / "summary.md"
    assert page.main(_argv(tmp_path, "check", summary=summary), git=FakeGit(), call=github) == 0, capsys.readouterr()
    assert [method for method, _, _ in github.calls] == ["GET"], "it never creates a page"
    text = summary.read_text(encoding="utf-8")
    assert "checked before the build" in text and page.PLACEHOLDER_DIGEST in text
    assert "**This release changes the database.** It adds 1 migration," in text


REFUSALS = [
    (dict(shallow=True), {}, "0.11.0\n", "the checkout is shallow"),
    (dict(tags=("v0.11.0",)), {}, "0.11.0\n", "there is no tag v0.12.0 in the checkout"),
    ({}, {}, None, "the upgrade file upgrade.txt is missing"),
    ({}, {}, "0.11\n", "the upgrade file's value is not a version"),
    ({}, {}, "none\n", "the upgrade file says none, but v0.11.0 is an earlier release"),
    ({}, {}, "0.10.0\n", "there is no tag v0.10.0"),
    (dict(new={**OLD_TREE, "r3": "revision = R3\ndown_revision = 'r2'\n"}), {}, "0.11.0\n",
     "r3.py: revision is not a non-empty string literal"),
    ({}, dict(existing=200), "0.11.0\n", "it already has a release page, which is never replaced"),
    ({}, dict(existing=500), "0.11.0\n", "GitHub answered 500"),
]


@pytest.mark.parametrize("git,github,upgrade,message", REFUSALS)
@pytest.mark.parametrize("command", ["check", "publish"])
def test_the_early_check_refuses_whatever_the_final_run_refuses(tmp_path, capsys, command, git, github, upgrade,
                                                                 message):
    fake = FakeGitHub(**github)
    assert page.main(_argv(tmp_path, command, upgrade=upgrade), git=FakeGit(**git), call=fake) == 1
    assert message in capsys.readouterr().out
    assert not [call for call in fake.calls if call[0] == "POST"]
    assert not (tmp_path / "notes.md").exists()


def test_the_check_runs_in_the_changes_job_before_anything_builds():
    steps = JOBS["changes"]["steps"]
    names = [step.get("name") for step in steps]
    checkout = steps[names.index("Checkout (the history the release page is written from; v* tags only)")]
    check = steps[names.index("Release page check (v* tags only)")]
    preflight = names.index("Release preflight (v* tags only)")
    pick = next(i for i, step in enumerate(steps) if step.get("id") == "pick")
    assert preflight < steps.index(checkout) < steps.index(check) < pick
    assert JOBS["build"]["needs"] == "changes", "every build waits for the job the check is in"
    assert checkout["if"] == check["if"] == ON_A_TAG
    assert checkout["uses"] == steps[0]["uses"], "the pin the job already uses"
    assert checkout["with"] == {"persist-credentials": False, "fetch-depth": 0, "path": "release-history"}
    assert check["working-directory"] == "release-history"
    assert check["env"] == {"GH_TOKEN": "${{ github.token }}", "VERSION": "${{ steps.release.outputs.version }}"}
    assert check["run"].startswith('python3 scripts/ci/release_page.py check --repo "$GITHUB_REPOSITORY"')
    assert "${{" not in check["run"] and "--digests" not in check["run"]
    named = re.findall(r"--image (\S+)", check["run"])
    assert named == [entry["name"] for entry in JOBS["merge"]["strategy"]["matrix"]["image"]]
    assert JOBS["changes"]["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "read"}, \
        "a read-only token: it cannot create a release"


# --------------------------------------------------------------------------- the create call


def test_the_create_call_names_the_tagged_commit(tmp_path, capsys):
    github = FakeGitHub()
    assert page.main(_argv(tmp_path, "publish"), git=FakeGit(), call=github) == 0, capsys.readouterr().out
    method, path, payload = github.calls[-1]
    assert (method, path) == ("POST", f"/repos/{REPO}/releases")
    assert payload["tag_name"] == "v0.12.0" and payload["target_commitish"] == SHA


def test_a_summary_that_cannot_be_written_after_the_page_is_created_is_a_warning(tmp_path, capsys):
    github = FakeGitHub()
    unwritable = tmp_path / "a-directory"
    unwritable.mkdir()
    assert page.main(_argv(tmp_path, "publish", summary=unwritable), git=FakeGit(), call=github) == 0
    out = capsys.readouterr().out
    assert "::warning title=Release page::the step summary could not be written" in out
    assert "::error" not in out and github.calls[-1][0] == "POST"


# --------------------------------------------------------------------------- annotations


def test_an_annotation_is_escaped_so_it_cannot_start_another_command():
    line = page.annotation("error", "a%b\r\n::warning title=x::y")
    assert line == "::error title=Release page::a%25b%0D%0A::warning title=x::y"
    assert page.escaped("a:b,c", property_value=True) == "a%3Ab%2Cc"


def test_a_path_holding_a_line_break_stays_on_the_annotations_one_line(tmp_path, capsys):
    stem = "odd\n::add-mask::x\n::warning title=x::y"
    new = {**OLD_TREE, stem: "revision = R3\ndown_revision = 'r2'\n"}
    assert page.main(_argv(tmp_path, "check"), git=FakeGit(new=new), call=FakeGitHub()) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("::error title=Release page::")
    assert "odd%0A::add-mask::x%0A::warning title=x::y.py" in lines[0]


# --------------------------------------------------------------------------- a commit with no message


def test_a_commit_with_an_empty_message_is_listed_by_its_commit():
    commit = "e" * 40
    changes = history.parse_log(f"{commit}\n{'1' * 40}\n\n\0{'f' * 40}\n{'1' * 40}\nfix: kept (#2)\n\0")
    assert [(change.commit, change.text) for change in changes] == [(commit, ""), ("f" * 40, "kept")]
    notes = notes_md.Notes(repository=REPO, version="0.12.0", sha=SHA, previous="v0.11.0", changes=tuple(changes),
                           database=history.database_change({}, {}), upgrade_from="0.11.0",
                           images=(("api", DIGEST),), chart_digest=DIGEST)
    line = notes_md.change_line(changes[0], notes)
    assert line == f"- (a commit with no subject) ([{commit[:12]}](https://github.com/{REPO}/commit/{commit}))"
    assert line in notes_md.render(notes)
