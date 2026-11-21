"""PRD-279 US-301 and US-302: the release page a version tag creates, and what it says.

* **The previous release** is the highest full release below the tag's version among the
  tags reachable from the tagged commit, by version order: for a release, a candidate, a
  patch on its release branch, and the first release (none).
* **What changed** is the merged pull requests since then, from squash merges and merge
  commits alike, grouped by type. Their text is data: whatever a title holds, each shows
  as one code span in one list item, and every link on the page is this repository's.
* **Database** compares the Alembic revisions at the two tags: a release that adds a
  migration, one that adds none, one that removes or edits a shipped migration (never
  "does not change"), and the first release. The PRD's sentence on going back is there.
* **Upgrading** reads one file, and a missing or malformed file stops the job.
* **The page** has the five sections of §5.4 in order, a candidate is a pre-release, the
  checking page is linked, and a page that exists is never replaced.
* **The job** is the only one that may write contents, holds nothing else, runs last on a
  tag only, and puts nothing from a tag, a commit or a file into its shell.

Fixtures and fakes stand in for git and GitHub: no network, and no git history of its own.
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
OLD = "1111111111111111111111111111111111111111"
DIGEST = "sha256:" + "a" * 64
CHART = "sha256:" + "c" * 64
NAMES = ["api", "frontend", "enterprise-frontend", "workspace-worker"]
VERSIONS = "orchestrator/alembic/versions"
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text(encoding="utf-8"))
JOBS = WORKFLOW["jobs"]
ON_A_TAG = "startsWith(github.ref, 'refs/tags/v')"
# The Arabic letter mark, the soft hyphen, the combining grapheme joiner, the Mongolian vowel
# separator, the interlinear annotation characters, two tag characters, two variation selectors,
# a zero-width space, the line separator, a direction override and isolate, and the byte order mark.
INVISIBLE_SAMPLE = (0x061C, 0x00AD, 0x034F, 0x180E, 0xFFF9, 0xFFFA, 0xFFFB, 0xE0001, 0xE0041, 0xFE0F, 0xE0100,
                    0x200B, 0x2028, 0x202E, 0x2066, 0xFEFF)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


history = _load("release_history")
notes_md = _load("release_notes")
page = _load("release_page")
signing = _load("release_signing")


def _source(revision: str, down) -> str:
    return f'"""A migration."""\nrevision = {revision!r}\ndown_revision = {down!r}\n\ndef upgrade():\n    pass\n'


def _revs(*pairs) -> dict:
    return history.revisions({f"{VERSIONS}/{rev}.py": _source(rev, down) for rev, down in pairs})


BASE = (("r1", None), ("r2", "r1"))


def _notes(**overrides) -> "notes_md.Notes":
    values = dict(repository=REPO, version="0.12.0", sha=SHA, previous="v0.11.0", changes=(),
                  database=history.database_change(_revs(*BASE), _revs(*BASE)), upgrade_from="0.11.0",
                  images=tuple((name, DIGEST) for name in NAMES), chart_digest=CHART)
    values.update(overrides)
    return notes_md.Notes(**values)


# --------------------------------------------------------------------------- the previous release


@pytest.mark.parametrize("version,reachable,expected", [
    # A release: the highest full release below it, by version order (0.10 is above 0.9).
    ("0.12.0", ["v0.9.0", "v0.10.0", "v0.11.0", "v0.11.1"], "v0.11.1"),
    ("0.11.0", ["v0.9.0", "v0.10.0"], "v0.10.0"),
    # Its own tag and its candidates are reachable from its commit, and are not it.
    ("0.12.0", ["v0.11.1", "v0.12.0-rc.1", "v0.12.0-rc.2", "v0.12.0"], "v0.11.1"),
    # A candidate: the full release before it, never an earlier candidate.
    ("0.12.0-rc.2", ["v0.11.1", "v0.12.0-rc.1"], "v0.11.1"),
    # A patch on release/0.11: only that branch's tags are reachable from its commit.
    ("0.11.2", ["v0.11.0", "v0.11.1", "v0.11.2"], "v0.11.1"),
    ("0.11.2-rc.1", ["v0.10.3", "v0.11.0", "v0.11.1"], "v0.11.1"),
    # The first release: no full release before it. Tags of other forms are not releases.
    ("0.12.0", [], None),
    ("0.12.0", ["v0.12.0-rc.1", "archive/prd263-1e-original", "vnext", "v1.0"], None),
])
def test_the_previous_release_is_the_highest_full_release_below_it_that_its_commit_reaches(version, reachable, expected):
    assert history.previous_release(version, reachable) == expected


def test_a_patch_never_compares_with_a_newer_release_on_main():
    """main's v0.12.0 is not reachable from release/0.11's commit, and is above the patch anyway."""
    assert history.previous_release("0.11.2", ["v0.11.0", "v0.11.1", "v0.12.0"]) == "v0.11.1"


@pytest.mark.parametrize("version,tags,latest", [
    ("0.12.0", ["v0.11.0", "v0.11.1", "v0.12.0"], True),
    ("0.11.2", ["v0.11.1", "v0.12.0", "v0.11.2"], False),
    ("0.13.0-rc.1", ["v0.12.0"], False),
    ("0.12.0", [], True),
])
def test_only_the_highest_full_release_is_marked_latest(version, tags, latest):
    assert history.is_latest(version, tags) is latest


# --------------------------------------------------------------------------- what changed


def _log(*commits) -> str:
    return "".join(f"{sha}\n{parents}\n{subject}\n{body}\0" for sha, parents, subject, body in commits)


LOG = _log(
    ("a" * 40, f"{OLD} {'b' * 40}", "Merge pull request #41 from someone/branch", "\nfix(chat): keep the draft\n"),
    ("c" * 40, OLD, "feat(prd-279)!: the release page (#40)", "Body.\n\nBREAKING CHANGE: a setting is gone"),
    ("d" * 40, OLD, "docs: the checking page (#39)", ""),
    ("e" * 40, OLD, "Update README.md", ""),
    ("f" * 40, OLD, "wip: half a thing (#38)", "BREAKING CHANGE: maybe"),
)


def test_each_merged_pull_request_is_read_by_its_title_and_grouped_by_its_type():
    changes = history.parse_log(LOG)
    assert [(c.kind, c.scope, c.text, c.number, c.breaking) for c in changes] == [
        ("fix", "chat", "keep the draft", 41, False),
        ("feat", "prd-279", "the release page", 40, True),
        ("docs", "", "the checking page", 39, False),
        ("other", "", "Update README.md", None, False),
        ("other", "", "wip: half a thing", 38, True),
    ]
    assert [heading for heading, _ in history.grouped(changes)] == ["Features", "Fixes", "Documentation",
                                                                   "Other changes"]


HOSTILE = [
    "<img src=x onerror=alert(1)> <script>alert(1)</script>",
    "# A heading\n## and another",
    "[a link](https://example.com/steal) ![an image](https://example.com/x.png) https://example.com",
    "@everyone and @AutomatosAI/maintainers, see #1 and GH-2",
    "close the span ` then **bold** and <b>html</b> `` twice",
    "a line\nthen\r\nmore" + chr(0x2028) + "and" + chr(0x202E) + "more\x00",
    "| a | table | row |",
    "x" * 1000,
    "",
    # Invisible and direction-changing characters, by code point (this file holds none of them).
    "before" + "".join(chr(point) for point in INVISIBLE_SAMPLE) + "after",
]


@pytest.mark.parametrize("text", HOSTILE)
def test_text_from_a_commit_is_one_code_span_that_cannot_break_out(text):
    shown = notes_md.literal(text)
    assert shown.startswith("`") and shown.endswith("`") and shown.count("`") == 2
    assert "\n" not in shown and "\r" not in shown and not set(shown) & {chr(point) for point in INVISIBLE_SAMPLE}
    assert len(shown) <= notes_md.TEXT_LIMIT + 2


@pytest.mark.parametrize("text", HOSTILE)
def test_a_hostile_title_stays_inside_its_list_item_and_links_nowhere_else(text):
    text = text.replace("\0", "")  # git keeps no NUL in a message; its log uses NUL between commits
    subject = f"feat({text[:20] or 'x'}): {text}".replace("\n", " ") + " (#7)"
    raw = _log(("a" * 40, OLD, subject, text))
    body = notes_md.render(_notes(changes=tuple(history.parse_log(raw))))
    section = body.split("## What changed", 1)[1].split("## Database", 1)[0]
    items = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(items) == 1 and items[0].count("`") == 2
    assert re.fullmatch(r"- (\*\*Breaking:\*\* )?`[^`\n]*` \(\[#7\]\(https://github\.com/AutomatosAI/automatos-ai/pull/7\)\)",
                        items[0]), items[0]
    _only_own_links(body)


def _only_own_links(body: str) -> None:
    outside = re.sub(r"`[^`]*`", "", re.sub(r"```.*?```", "", body, flags=re.S))
    for target in re.findall(r"\]\(([^)]*)\)", outside):
        assert target.startswith(f"https://github.com/{REPO}/"), target
    assert "<" not in outside.replace("<your Secret>", ""), "no HTML outside a code span"
    headings = [line for line in body.splitlines() if line.startswith("#")]
    assert all(re.fullmatch(r"#{2,3} [A-Z][A-Za-z ]+", line) for line in headings), headings


def test_a_long_history_is_cut_with_a_link_to_the_rest():
    raw = _log(*((f"{i:040x}", OLD, f"fix: change {i} (#{i + 1})", "") for i in range(notes_md.MAX_CHANGES + 5)))
    body = notes_md.render(_notes(changes=tuple(history.parse_log(raw))))
    assert body.count("\n- `") == notes_md.MAX_CHANGES
    assert "And 5 more: see [v0.11.0...v0.12.0](https://github.com/AutomatosAI/automatos-ai/compare/v0.11.0...v0.12.0)." in body


# --------------------------------------------------------------------------- the database


def test_a_release_that_adds_a_migration_says_it_changes_the_database():
    before, after = _revs(*BASE), _revs(*BASE, ("r4", "r3"), ("r3", "r2"))
    change = history.database_change(before, after)
    assert change.verdict == "changes"
    assert [rev.revision for rev in change.added] == ["r3", "r4"], "in the order they apply"
    assert (change.heads_before, change.heads_after) == (("r2",), ("r4",))
    text = "\n".join(notes_md.database_section(_notes(database=change)))
    assert "**This release changes the database.** It adds 2 migrations" in text
    assert "- `r3` in `r3.py`\n- `r4` in `r4.py`" in text
    assert notes_md.BACKUP in text


def test_a_release_with_no_new_migration_says_it_does_not():
    change = history.database_change(_revs(*BASE), _revs(*BASE))
    assert change.verdict == "none" and not change.added
    text = "\n".join(notes_md.database_section(_notes(database=change)))
    assert "**This release does not change the database.**" in text
    assert "changes the database." not in text.replace(notes_md.BACKUP, "")


@pytest.mark.parametrize("after", [
    {f"{VERSIONS}/r2.py": _source("r2", "r1")},                                          # r1 removed
    {f"{VERSIONS}/r1.py": _source("r1", None), f"{VERSIONS}/r2.py": _source("r2", "r1") + "# edited\n"},
])
def test_a_shipped_migration_removed_or_edited_is_never_no_change(after):
    change = history.database_change(_revs(*BASE), history.revisions(after))
    assert change.verdict == "edits" and not change.added
    text = "\n".join(notes_md.database_section(_notes(database=change)))
    assert "does not change the database" not in text
    assert "**This release changes the database.** It adds no migration" in text
    assert ("Removed since v0.11.0:" in text) or ("Edited since v0.11.0:" in text)


def test_one_added_and_another_removed_lists_both():
    change = history.database_change(_revs(*BASE), _revs(("r1", None), ("r3", "r1")))
    assert change.verdict == "changes"
    assert [r.revision for r in change.added] == ["r3"] and [r.revision for r in change.removed] == ["r2"]


def test_the_first_release_says_where_a_new_install_starts():
    change = history.database_change(None, _revs(*BASE))
    assert change.verdict == "first"
    text = "\n".join(notes_md.database_section(_notes(previous=None, database=change)))
    assert "**This is the first release.** A new install creates the database at `r2`, through 2 migrations." in text
    assert "- `r1`" not in text, "a first release does not list the whole history"


def test_a_merge_revision_is_named_with_what_it_joins():
    after = _revs(*BASE, ("r3", "r1"), ("m", ("r2", "r3")))
    change = history.database_change(_revs(*BASE), after)
    assert [r.revision for r in change.added] == ["r3", "m"] and change.heads_after == ("m",)
    assert "- `m` in `m.py` (merges `r2`, `r3`)" in "\n".join(notes_md.database_section(_notes(database=change)))


def test_revisions_are_read_as_text_never_run():
    files = {f"{VERSIONS}/r1.py": "import os\nos.system('exit 3')\nrevision = 'r1'\ndown_revision = None\n",
             f"{VERSIONS}/notes.py": "HELPER = 1\n",
             f"{VERSIONS}/typed.py": "revision: str = 'r2'\ndown_revision: str = 'r1'\n"}
    assert {rev: r.parents for rev, r in history.revisions(files).items()} == {"r1": (), "r2": ("r1",)}
    with pytest.raises(history.HistoryError, match="both declare 'r1'"):
        history.revisions({**files, f"{VERSIONS}/again.py": "revision = 'r1'\ndown_revision = None\n"})
    with pytest.raises(history.HistoryError, match="is not Python"):
        history.revisions({f"{VERSIONS}/bad.py": "revision = ("})


def test_the_versions_folder_is_read_at_each_tag_with_git_never_imported():
    git = FakeGit()
    files = page.revision_files(git, "refs/tags/v0.11.0")
    assert sorted(files) == [f"{VERSIONS}/r1.py", f"{VERSIONS}/r2.py"]
    assert ["ls-tree", "-z", "refs/tags/v0.11.0", "--", f"{VERSIONS}/"] in git.calls
    assert ["show", f"refs/tags/v0.11.0:{VERSIONS}/r1.py"] in git.calls


# --------------------------------------------------------------------------- the upgrade file


@pytest.mark.parametrize("text,expected", [
    ("none\n", None), ("# a comment\n\n0.11.0\n", "0.11.0"), ("  0.9.3  \n# after\n", "0.9.3"),
])
def test_the_upgrade_file_holds_one_version_or_none(text, expected):
    assert history.read_upgrade_from(text) == expected


@pytest.mark.parametrize("text", ["", "# only a comment\n", "0.11.0\n0.10.0\n", "0.11\n", "v0.11.0\n",
                                  "0.11.0-rc.1\n", "latest\n", "None\n"])
def test_a_malformed_upgrade_file_is_refused(text):
    with pytest.raises(history.HistoryError, match="upgrade file"):
        history.read_upgrade_from(text)


@pytest.mark.parametrize("value,match", [("0.12.0", "not below"), ("0.13.0", "not below"), ("0.10.0", "no tag v0.10.0")])
def test_the_upgrade_file_names_an_earlier_tagged_release(value, match):
    with pytest.raises(history.HistoryError, match=match):
        history.check_upgrade_from(value, "0.12.0", ["v0.11.0", "v0.12.0-rc.1"], "v0.11.0")
    history.check_upgrade_from("0.11.0", "0.12.0", ["v0.11.0"], "v0.11.0")


@pytest.mark.parametrize("version,previous", [("0.12.0", "v0.11.0"), ("0.11.3", "v0.11.2"), ("0.12.0-rc.1", "v0.11.0")])
def test_none_is_refused_once_there_is_an_earlier_release(version, previous):
    with pytest.raises(history.HistoryError, match=f"the upgrade file says none, but {previous} is an earlier release"):
        history.check_upgrade_from(None, version, [previous], previous)
    history.check_upgrade_from(None, "0.11.0", [], None)  # the first release


def test_the_repositorys_upgrade_file_is_well_formed():
    text = (ROOT / "scripts" / "ci" / "release_upgrade_from.txt").read_text(encoding="utf-8")
    assert page.UPGRADE_FILE.name == "release_upgrade_from.txt"
    value = history.read_upgrade_from(text)
    assert value is None or re.fullmatch(r"\d+\.\d+\.\d+", value)


@pytest.mark.parametrize("value,previous,line", [
    ("0.11.0", "v0.11.0", "An install of 0.11.0 or any later release upgrades to 0.12.0 in one step."),
    (None, None, "This is the first release, so there is nothing to upgrade from: install it new."),
])
def test_the_page_prints_the_upgrade_files_value(value, previous, line):
    text = "\n".join(notes_md.upgrade_section(_notes(upgrade_from=value, previous=previous)))
    assert line in text and "scripts/ci/release_upgrade_from.txt" in text


# --------------------------------------------------------------------------- the page


def test_the_page_has_the_five_sections_in_order_and_links_the_checking_page():
    body = notes_md.render(_notes(changes=tuple(history.parse_log(LOG))))
    assert [line for line in body.splitlines() if line.startswith("## ")] == [
        "## What changed", "## Database", "## Upgrading", "## What to install", "## Checking it"]
    link = "https://github.com/AutomatosAI/automatos-ai/blob/v0.12.0/docs/deployment-infrastructure/checking-a-release.md"
    assert f"[Checking a release]({link})" in body
    assert (ROOT / notes_md.CHECKING_PAGE).is_file()
    _only_own_links(body)


def test_what_to_install_names_the_chart_and_every_image_by_digest():
    body = notes_md.render(_notes())
    assert f"The chart `oci://ghcr.io/automatosai/charts/automatos`, version `0.12.0`, digest `{CHART}`." in body
    assert "helm install automatos oci://ghcr.io/automatosai/charts/automatos --version 0.12.0" in body
    for name in NAMES:
        assert f"| `ghcr.io/automatosai/automatos-{name}:0.12.0` | `{DIGEST}` |" in body


def test_checking_it_is_the_workflows_own_check_for_each_artifact():
    body = notes_md.render(_notes())
    signer = signing.release_signer(REPO, "refs/tags/v0.12.0", SHA)
    for reference in [f"ghcr.io/automatosai/charts/automatos@{CHART}"] + [
            f"ghcr.io/automatosai/automatos-{name}@{DIGEST}" for name in NAMES]:
        args = signing.verify_args(reference, signer)
        assert " ".join(args) in body.replace(" \\\n  ", " "), reference


@pytest.mark.parametrize("field,value", [("repository", "Automatos AI/x"), ("sha", "HEAD"), ("chart_digest", "latest"),
                                         ("images", (("api", "sha256:short"),)), ("previous", "v0.13.0"),
                                         ("version", "0.12"), ("sha", SHA + "\n"), ("upgrade_from", None)])
def test_a_value_shown_outside_a_code_span_must_be_of_its_form(field, value):
    with pytest.raises((notes_md.NotesError, ValueError)):
        notes_md.render(_notes(**{field: value}))


@pytest.mark.parametrize("version,prerelease,latest", [("0.12.0", False, "true"), ("0.12.0-rc.1", True, "false")])
def test_a_release_candidate_is_a_pre_release_and_never_the_latest(version, prerelease, latest):
    request = page.release_request(_notes(version=version), "body", latest=True)
    assert request == {"tag_name": f"v{version}", "target_commitish": SHA, "name": f"Automatos {version}",
                       "body": "body", "draft": False, "prerelease": prerelease, "make_latest": latest}


class FakeGitHub:
    def __init__(self, existing: int = 404, created: int = 201):
        self.existing, self.created, self.calls = existing, created, []

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if method == "GET":
            return self.existing, None
        return self.created, {"html_url": f"https://github.com/{REPO}/releases/tag/v0.12.0"}


def test_the_page_is_created_once_and_never_over_one_that_exists():
    github = FakeGitHub()
    assert page.publish(_notes(), "body", True, github).endswith("/releases/tag/v0.12.0")
    assert [(m, p) for m, p, _ in github.calls] == [("GET", f"/repos/{REPO}/releases/tags/v0.12.0"),
                                                   ("POST", f"/repos/{REPO}/releases")]
    for existing, reason in ((200, "never replaced"), (500, "GitHub answered 500"), (403, "GitHub answered 403")):
        github = FakeGitHub(existing=existing)
        with pytest.raises(page.PageError, match=reason):
            page.publish(_notes(), "body", True, github)
        assert [m for m, _, _ in github.calls] == ["GET"], "nothing is written"
    with pytest.raises(page.PageError, match="refused"):
        page.publish(_notes(), "body", True, FakeGitHub(created=422))


# --------------------------------------------------------------------------- the command, end to end


class FakeGit:
    """A repository with v0.11.0 (r1, r2) and the tagged commit (r1, r2, r3)."""

    TAGS = ["v0.11.0", "v0.12.0-rc.1", "v0.12.0"]
    TREES = {"refs/tags/v0.11.0": BASE, SHA: BASE + (("r3", "r2"),)}

    def __init__(self, tags=None, shallow: bool = False, trees=None):
        self.tags, self.calls = tags if tags is not None else self.TAGS, []
        self.shallow, self.trees = shallow, trees if trees is not None else self.TREES

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if args == ["rev-parse", "--is-shallow-repository"]:
            return "true\n" if self.shallow else "false\n"
        if args[:2] == ["tag", "--list"]:
            return "\n".join(self.tags) + "\n"
        if args[0] == "ls-tree":
            files = self.trees[args[2]]
            return "".join(f"100644 blob {'0' * 40}\t{VERSIONS}/{rev}.py\0" for rev in self._sources(files)) + \
                f"040000 tree {'0' * 40}\t{VERSIONS}/__pycache__\0"
        if args[0] == "show":
            ref, path = args[1].split(":", 1)
            return self._sources(self.trees[ref])[Path(path).stem]
        if args[0] == "log":
            assert args[-1] == f"refs/tags/v0.11.0..{SHA}"
            return LOG
        raise AssertionError(args)

    @staticmethod
    def _sources(tree) -> dict:
        """``{stem: source}``: a tree is ``((revision, down), …)`` pairs or ``{stem: source}`` as written."""
        return dict(tree) if isinstance(tree, dict) else {rev: _source(rev, down) for rev, down in tree}


def _argv(tmp_path: Path, upgrade: str = "0.11.0\n", version: str = "0.12.0") -> list:
    digests = tmp_path / "digests"
    digests.mkdir(exist_ok=True)
    for name in NAMES:
        (digests / name).write_text(DIGEST + "\n", encoding="utf-8")
    argv = ["publish", "--repo", REPO, "--version", version, "--sha", SHA, "--digests", str(digests),
            "--chart-digest", CHART, "--notes", str(tmp_path / "notes.md")] + [
                arg for name in NAMES for arg in ("--image", name)]
    if upgrade is not None:
        (tmp_path / "upgrade.txt").write_text(upgrade, encoding="utf-8")
    return argv + ["--upgrade-file", str(tmp_path / "upgrade.txt")]


def test_the_command_writes_the_notes_and_creates_the_page(tmp_path, capsys):
    github = FakeGitHub()
    assert page.main(_argv(tmp_path), git=FakeGit(), call=github) == 0, capsys.readouterr().out
    body = (tmp_path / "notes.md").read_text(encoding="utf-8")
    assert "Since v0.11.0, by pull request" in body and "**This release changes the database.** It adds 1 migration," in body
    assert "- `r3` in `r3.py`" in body
    assert github.calls[-1][2]["body"] == body and github.calls[-1][2]["make_latest"] == "true"


@pytest.mark.parametrize("upgrade,message", [
    (None, "the upgrade file upgrade.txt is missing"),
    ("0.11\n", "the upgrade file's value is not a version"),
    ("0.11.0\n0.10.0\n", "must hold one value"),
    ("0.10.0\n", "there is no tag v0.10.0"),
])
def test_a_missing_or_malformed_upgrade_file_stops_the_job_before_anything_is_written(tmp_path, capsys, upgrade, message):
    github = FakeGitHub()
    assert page.main(_argv(tmp_path, upgrade=upgrade), git=FakeGit(), call=github) == 1
    assert message in capsys.readouterr().out
    assert not github.calls and not (tmp_path / "notes.md").exists()


def test_a_tag_the_checkout_does_not_have_stops_the_job(tmp_path, capsys):
    assert page.main(_argv(tmp_path), git=FakeGit(tags=["v0.11.0"]), call=FakeGitHub()) == 1
    assert "there is no tag v0.12.0 in the checkout" in capsys.readouterr().out


# --------------------------------------------------------------------------- the job


def test_the_release_page_job_runs_last_on_a_tag_and_may_write_contents_only():
    job = JOBS["release-notes"]
    assert job["if"] == ON_A_TAG
    assert job["needs"] == ["changes", "release-images", "release-chart"]
    assert job["permissions"] == {"contents": "write"}, "no packages, no OIDC token: it cannot sign or publish"
    writers = [name for name, other in JOBS.items() if (other.get("permissions") or {}).get("contents") == "write"]
    assert writers == ["release-notes"]
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert "secrets." not in yaml.safe_dump(job)
    assert JOBS["release-chart"]["outputs"] == {"digest": "${{ steps.chart.outputs.digest }}"}
    assert job["env"] == {"VERSION": "${{ needs.changes.outputs.version }}",
                          "CHART_DIGEST": "${{ needs.release-chart.outputs.digest }}"}


def test_the_job_reads_the_full_history_and_puts_nothing_into_its_shell():
    steps = JOBS["release-notes"]["steps"]
    checkout = steps[0]
    assert checkout["with"] == {"persist-credentials": False, "fetch-depth": 0}
    run = next(step for step in steps if "run" in step)
    assert "${{" not in run["run"]
    assert run["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert run["run"].startswith(
        'python3 scripts/ci/release_page.py publish --repo "$GITHUB_REPOSITORY" --version "$VERSION"')
    assert '--sha "$GITHUB_SHA"' in run["run"] and '--chart-digest "$CHART_DIGEST"' in run["run"]
    named = re.findall(r"--image (\S+)", run["run"])
    assert named == [entry["name"] for entry in JOBS["merge"]["strategy"]["matrix"]["image"]]


# --------------------------------------------------------------------------- the checking page (US-302)


def test_the_checking_page_gives_the_command_the_identity_and_what_a_check_proves():
    text = (ROOT / notes_md.CHECKING_PAGE).read_text(encoding="utf-8")
    for flag in signing.CLAIM_FLAGS:
        assert flag in text, flag
    for part in ("https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@refs/tags/v",
                 signing.ISSUER, "`AutomatosAI/automatos-ai`", "`images.yml`", "refs/tags/v<version>",
                 "ghcr.io/automatosai/charts/automatos@sha256:", "ghcr.io/automatosai/automatos-api@sha256:",
                 "The contents list", "The build record", ".SBOM", ".Provenance",
                 "A passing check proves", "A passing check does not prove"):
        assert part in text, part
    assert "checking-a-release.md" in (ROOT / "docs" / "deployment-infrastructure" / "releases.md").read_text(
        encoding="utf-8")
