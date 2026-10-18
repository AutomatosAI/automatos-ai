"""PRD-262 US-003: ``edge`` moves only after the published images booted.

``scripts/ci/promote_edge.py`` runs in smoke-images.yml's promote-edge job. These tests
fake the registry, GitHub's compare API and ``docker buildx``:

* ``edge`` moves when this commit is ahead of the one ``edge`` was built from (or the
  same), and stays when it is behind: a re-run of an old smoke never moves it back;
* a diverged history or an ``edge`` with no revision label stops the move;
* the move is ``imagetools create -t …:edge …:sha-<7>`` with one source and no
  annotation, so ``edge`` keeps the signed digest, and it is read back;
* every image is planned before any moves: one image that cannot be read leaves all
  three ``edge`` tags where they were.
"""
from __future__ import annotations

import http.client
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
OLD = "697ca7d6e" + "0" * 31
PREFIX = "ghcr.io/automatosai/automatos-"
DIGEST = {name: "sha256:" + char * 64 for name, char in (("api", "a"), ("frontend", "b"), ("workspace-worker", "c"))}
EDGE_DIGEST = "sha256:" + "e" * 64


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


promote = _load("promote_edge")


class FakeGHCR:
    """Tags to digests, and each ``edge``'s revision label; ``imagetools create`` moves a tag."""

    def __init__(self, edge_revision=OLD, missing=()):
        self.tags = {f"{PREFIX}{name}:sha-0d9f0ac": digest for name, digest in DIGEST.items()}
        self.tags.update({f"{PREFIX}{name}:edge": EDGE_DIGEST for name in DIGEST})
        self.edge_revision = edge_revision
        self.missing = set(missing)
        self.commands = []

    def resolve(self, ref):
        if ref in self.missing or ref not in self.tags:
            raise urllib.error.HTTPError(ref, 404, "manifest unknown", {}, io.BytesIO())
        return self.tags[ref]

    def image(self, ref, arch):
        if self.edge_revision is None:
            raise urllib.error.HTTPError(ref, 404, "manifest unknown", {}, io.BytesIO())
        labels = {promote.REVISION_LABEL: self.edge_revision} if self.edge_revision else {}
        return {"layers": []}, {"config": {"Labels": labels}}, None

    def run(self, command, check):
        assert check is True
        self.commands.append(command)
        target, source = command[-2], command[-1]
        self.tags[target] = self.tags[source]


def _install(monkeypatch, ghcr: FakeGHCR, status="ahead"):
    compared = []
    monkeypatch.setattr(promote.registry, "resolve", ghcr.resolve)
    monkeypatch.setattr(promote.registry, "image", ghcr.image)
    monkeypatch.setattr(promote, "compare", lambda repo, base, head: compared.append((repo, base, head)) or status)
    return compared


def _args(tmp_path, images=("api", "frontend", "workspace-worker")):
    argv = ["--sha", SHA, "--tag", "sha-0d9f0ac", "--repo", "AutomatosAI/automatos-ai", "--prefix", PREFIX,
            "--summary", str(tmp_path / "summary.md")]
    for name in images:
        argv += ["--image", name]
    return promote.build_parser().parse_args(argv)


@pytest.mark.parametrize("status", ["ahead", "identical"])
def test_edge_moves_to_a_newer_or_the_same_commit_keeping_the_signed_digest(monkeypatch, tmp_path, status):
    ghcr = FakeGHCR()
    compared = _install(monkeypatch, ghcr, status)
    assert promote.promote(_args(tmp_path), run=ghcr.run) == 0
    assert compared == [("AutomatosAI/automatos-ai", OLD, SHA)] * 3
    for name, digest in DIGEST.items():
        assert ghcr.tags[f"{PREFIX}{name}:edge"] == digest
    for command in ghcr.commands:
        assert command[:5] == ["docker", "buildx", "imagetools", "create", "-t"]
        assert len(command) == 7, "one source and no --annotation: imagetools copies the signed index as it is"
        assert command[5].endswith(":edge") and command[6].endswith(":sha-0d9f0ac")
    text = (tmp_path / "summary.md").read_text()
    assert "| api | `sha-0d9f0ac` | `sha256:aaaaaaaaaaaa…` | `697ca7d6e000` |" in text


def test_edge_stays_when_a_newer_commit_already_moved_it(monkeypatch, tmp_path):
    ghcr = FakeGHCR()
    _install(monkeypatch, ghcr, "behind")
    assert promote.promote(_args(tmp_path), run=ghcr.run) == 0
    assert ghcr.commands == []
    assert all(ghcr.tags[f"{PREFIX}{name}:edge"] == EDGE_DIGEST for name in DIGEST)
    assert "| keep |" in (tmp_path / "summary.md").read_text()


def test_the_first_edge_is_created_without_a_comparison(monkeypatch, tmp_path):
    ghcr = FakeGHCR(edge_revision=None)
    compared = _install(monkeypatch, ghcr)
    assert promote.promote(_args(tmp_path, ["api"]), run=ghcr.run) == 0
    assert compared == [] and len(ghcr.commands) == 1


@pytest.mark.parametrize("edge_revision,status", [(OLD, "diverged"), ("", "ahead")])
def test_an_order_that_cannot_be_told_leaves_edge_alone(monkeypatch, tmp_path, capsys, edge_revision, status):
    ghcr = FakeGHCR(edge_revision=edge_revision)
    _install(monkeypatch, ghcr, status)
    assert promote.promote(_args(tmp_path), run=ghcr.run) == 1
    assert ghcr.commands == []
    assert "::error title=promote-edge::edge stays where it is" in capsys.readouterr().out


def test_one_unreadable_image_moves_none(monkeypatch, tmp_path):
    ghcr = FakeGHCR(missing={f"{PREFIX}workspace-worker:sha-0d9f0ac"})
    _install(monkeypatch, ghcr)
    assert promote.promote(_args(tmp_path), run=ghcr.run) == 1
    assert ghcr.commands == [], "api and frontend must not move while the worker's tag is missing"


def test_a_registry_read_cut_short_leaves_edge_alone(monkeypatch, tmp_path, capsys):
    ghcr = FakeGHCR()
    _install(monkeypatch, ghcr)

    def cut_short(ref):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(promote.registry, "resolve", cut_short)
    assert promote.promote(_args(tmp_path), run=ghcr.run) == 1
    assert ghcr.commands == []
    assert "::error title=promote-edge::edge stays where it is" in capsys.readouterr().out


def test_a_move_that_lands_on_another_digest_fails(monkeypatch):
    ghcr = FakeGHCR()
    _install(monkeypatch, ghcr)
    step = promote.plan("api", PREFIX, "sha-0d9f0ac", SHA, "AutomatosAI/automatos-ai")
    with pytest.raises(RuntimeError):
        promote.move(step, run=lambda command, check: None)  # the tag never moved


@pytest.mark.parametrize("previous,status,action", [
    (None, None, "move"), (OLD, "ahead", "move"), (OLD, "identical", "move"), (OLD, "behind", "keep"),
])
def test_decide(previous, status, action):
    assert promote.decide(previous, status) == action


def test_compare_asks_github_for_the_order_of_the_two_commits(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        pass

    def urlopen(request, timeout):
        seen.update(url=request.full_url, auth=request.get_header("Authorization"))
        return Response(json.dumps({"status": "ahead"}).encode())

    monkeypatch.setenv("GH_TOKEN", "t0ken")
    monkeypatch.setattr(promote.urllib.request, "urlopen", urlopen)
    assert promote.compare("AutomatosAI/automatos-ai", OLD, SHA) == "ahead"
    assert seen["url"] == f"https://api.github.com/repos/AutomatosAI/automatos-ai/compare/{OLD}...{SHA}?per_page=1"
    assert seen["auth"] == "Bearer t0ken"
