"""PRD-279 FR-6 and §5.5: a release never replaces a version, and nothing is visible unproven.

* ``scripts/ci/release_preflight.py``: before a `v*` tag builds anything, a version any
  published image already has stops the run, as does a registry that does not answer,
  and so does a commit that is in neither the default branch nor ``release/MAJOR.MINOR``.
* ``scripts/ci/release_tags.py``: the version tags go on last, by digest, all images
  planned before any is tagged; a version on another digest is never moved, and a
  stopped run of the tag job can finish the set.
* ``scripts/ci/release_chart.py remove``: a chart pushed but not proven is deleted
  through GitHub's packages API, or the job says what a maintainer must delete.

Fakes stand in for the registry, GitHub and docker: no network, no Docker.
"""
from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "ghcr.io/automatosai/automatos-"
IMAGES = ("api", "frontend", "workspace-worker")
DIGESTS = {name: "sha256:" + letter * 64 for name, letter in zip(IMAGES, "abc")}
OTHER = "sha256:" + "f" * 64
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("release_preflight")
tags = _load("release_tags")
chart = _load("release_chart")


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://ghcr.io/v2/x", code, "answer", {}, None)


# --------------------------------------------------------------------------- the preflight


def _no_releases_yet(args) -> str:
    """git in a full-history checkout that holds no version tag."""
    return "false\n" if list(args) == ["rev-parse", "--is-shallow-repository"] else ""


def _preflight(version: str, state, compare) -> tuple:
    args = ["--version", version, "--sha", SHA, "--repo", "AutomatosAI/automatos-ai", "--default-branch", "main",
            "--prefix", PREFIX] + [arg for name in IMAGES for arg in ("--image", name)]
    return preflight.main(args, state=state, compare=compare, git=_no_releases_yet)


def test_a_new_version_on_main_may_build(capsys):
    assert _preflight("0.7.0", lambda ref: False, {"main": "ahead"}.get) == 0
    assert "::error" not in capsys.readouterr().out


def test_a_version_an_image_already_has_stops_the_run(capsys):
    taken = f"{PREFIX}frontend:0.7.0"
    assert _preflight("0.7.0", lambda ref: ref == taken, {"main": "identical"}.get) == 1
    out = capsys.readouterr().out
    assert f"{taken} is already published, and a version is never replaced: cut the next version" in out
    assert f"{PREFIX}api:0.7.0" not in out


def test_a_registry_that_does_not_answer_stops_the_run(capsys):
    assert _preflight("0.7.0", lambda ref: None, {"main": "ahead"}.get) == 1
    assert f"could not tell whether {PREFIX}api:0.7.0 is published, so nothing was built" in capsys.readouterr().out


@pytest.mark.parametrize("statuses,holder", [
    ({"main": "ahead"}, "main"),
    ({"main": "identical"}, "main"),
    ({"main": "diverged", "release/0.7": "ahead"}, "release/0.7"),
    ({"main": "behind", "release/0.7": "identical"}, "release/0.7"),
    ({"main": "diverged", "release/0.7": "diverged"}, None),
    ({"main": "behind"}, None),
])
def test_the_commit_must_be_on_main_or_its_own_release_branch(statuses, holder):
    branches = preflight.release_lines("0.7.1", "main")
    assert branches == ["main", "release/0.7"]
    assert preflight.line_holding(branches, statuses.get) == holder


def test_a_commit_on_neither_stops_the_run_and_a_candidate_uses_its_own_line(capsys):
    assert preflight.release_lines("0.8.0-rc.1", "main") == ["main", "release/0.8"]
    assert _preflight("0.8.0-rc.1", lambda ref: False, {"release/0.7": "ahead"}.get) == 1
    assert f"{SHA[:12]} is on neither main nor release/0.8" in capsys.readouterr().out


def test_a_compare_that_does_not_answer_stops_the_run(capsys):
    def broken(branch):
        raise _http(502)
    assert _preflight("0.7.0", lambda ref: False, broken) == 1
    assert "could not tell whether" in capsys.readouterr().out


@pytest.mark.parametrize("raised,expected", [(None, True), (_http(404), False), (_http(401), None),
                                             (urllib.error.URLError("down"), None)])
def test_the_registry_answer_is_read_as_published_absent_or_unknown(monkeypatch, raised, expected):
    def resolve(ref):
        if raised is not None:
            raise raised
        return DIGESTS["api"]
    monkeypatch.setattr(preflight.registry, "resolve", resolve)
    assert preflight.published(f"{PREFIX}api:0.7.0") is expected


# --------------------------------------------------------------------------- the version tags


class Registry:
    """Tags on a fake registry; ``run`` plays ``docker buildx imagetools create``."""

    def __init__(self, tags: dict | None = None, unreadable: tuple = ()):
        self.tags, self.unreadable, self.commands = dict(tags or {}), unreadable, []

    def lookup(self, ref: str):
        if ref in self.unreadable:
            raise tags.TagError(f"could not read {ref}")
        return self.tags.get(ref)

    def run(self, command, check):
        self.commands.append(command)
        source_digest = command[-1].split("@", 1)[1]
        self.tags = {**self.tags, **{command[i + 1]: source_digest for i, arg in enumerate(command) if arg == "-t"}}
        return SimpleNamespace(returncode=0)


def _main(tmp_path: Path, version: str, fake: Registry) -> int:
    for name, digest in DIGESTS.items():
        (tmp_path / name).write_text(digest + "\n", encoding="utf-8")
    args = ["--digests", str(tmp_path), "--prefix", PREFIX, "--version", version]
    args += [arg for name in IMAGES for arg in ("--image", name)]
    return tags.main(args, lookup=fake.lookup, run=fake.run)


def test_a_full_release_tags_every_image_with_the_version_and_its_line(tmp_path):
    fake = Registry()
    assert _main(tmp_path, "0.7.0", fake) == 0
    for name, digest in DIGESTS.items():
        assert fake.tags[f"{PREFIX}{name}:0.7.0"] == fake.tags[f"{PREFIX}{name}:0.7"] == digest
    assert all(command[-1].endswith(tuple(DIGESTS.values())) for command in fake.commands), "by digest"


def test_a_candidate_never_moves_the_release_line(tmp_path):
    fake = Registry()
    assert _main(tmp_path, "0.8.0-rc.1", fake) == 0
    assert sorted(fake.tags) == sorted(f"{PREFIX}{name}:0.8.0-rc.1" for name in IMAGES)


def test_a_version_on_another_digest_is_never_moved_and_no_image_is_tagged(tmp_path, capsys):
    fake = Registry({f"{PREFIX}workspace-worker:0.7.0": OTHER})
    assert _main(tmp_path, "0.7.0", fake) == 1
    assert fake.commands == [], "planned before any is tagged: the images never part ways"
    assert "already points at" in capsys.readouterr().out


def test_an_unreadable_tag_tags_nothing(tmp_path):
    fake = Registry(unreadable=(f"{PREFIX}api:0.7.0",))
    assert _main(tmp_path, "0.7.0", fake) == 1 and fake.commands == []


def test_a_stopped_run_finishes_the_set_from_the_same_digests(tmp_path):
    fake = Registry({f"{PREFIX}api:0.7.0": DIGESTS["api"]})
    assert _main(tmp_path, "0.7.0", fake) == 0
    api = next(command for command in fake.commands if command[-1].startswith(f"{PREFIX}api@"))
    assert f"{PREFIX}api:0.7.0" not in api and f"{PREFIX}api:0.7" in api, "only the line moves on api"
    assert fake.tags[f"{PREFIX}frontend:0.7.0"] == DIGESTS["frontend"]


def test_a_tag_that_lands_on_another_digest_fails():
    fake = Registry()
    plans = tags.plan(DIGESTS, PREFIX, "0.7.0", fake.lookup)

    def lying_lookup(ref):
        return OTHER if fake.commands else fake.lookup(ref)
    with pytest.raises(tags.TagError, match="after tagging"):
        tags.apply(plans, lying_lookup, fake.run)


def test_a_missing_digest_tags_nothing(tmp_path):
    with pytest.raises(tags.TagError, match="api: no signed digest"):
        tags.read_digests(tmp_path, ["api"])


# --------------------------------------------------------------------------- taking an unproven chart back


def _version(number: int, *labels: str) -> dict:
    return {"id": number, "name": f"sha256:{number:064x}", "metadata": {"container": {"tags": list(labels)}}}


def test_the_chart_package_is_named_for_an_organisation_or_a_user():
    assert chart.package_path("AutomatosAI", "Organization") == "/orgs/AutomatosAI/packages/container/charts%2Fautomatos"
    assert chart.package_path("someone", "User") == "/users/someone/packages/container/charts%2Fautomatos"


def test_only_the_version_this_job_pushed_is_deleted_or_the_package_when_it_is_the_only_one():
    package = chart.package_path("AutomatosAI", "Organization")
    versions = [_version(9, "0.8.0-rc.1"), _version(8, "sha256-abc.sig"), _version(7, "0.7.0")]
    assert chart.removal(package, versions, "0.8.0-rc.1") == f"{package}/versions/9"
    assert chart.removal(package, [_version(9, "0.7.0-rc.1")], "0.7.0-rc.1") == package
    with pytest.raises(chart.ChartError, match="0 versions"):
        chart.removal(package, versions, "0.9.0")


def test_remove_reads_the_versions_then_deletes_one():
    calls = []

    def call(method, path, token):
        calls.append((method, path, token))
        return [_version(9, "0.8.0-rc.1"), _version(7, "0.7.0")] if method == "GET" else None
    target = chart.remove("AutomatosAI", "Organization", "0.8.0-rc.1", "t", call=call)
    assert target.endswith("/versions/9")
    assert [method for method, _, _ in calls] == ["GET", "DELETE"] and all(token == "t" for *_, token in calls)


def test_a_refused_delete_says_what_a_maintainer_must_delete(monkeypatch, capsys):
    def refuse(method, path, token):
        raise _http(403)
    monkeypatch.setattr(chart, "_github", refuse)
    assert chart.main(["remove", "--owner", "AutomatosAI", "--owner-type", "Organization", "--tag", "0.8.0-rc.1"]) == 1
    out = capsys.readouterr().out
    assert "::error title=Unproven chart left published::ghcr.io/automatosai/charts/automatos:0.8.0-rc.1" in out
    assert "must delete that version of the charts/automatos package by hand" in out
