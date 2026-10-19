"""PRD-262 US-036: the tag `make up` picks, against a stubbed registry, one case per branch.

``scripts/stack_images.py`` decides which tag the three app images run and how each
gets there; these tests feed it a fake Docker and registry (``FakeLookup``) and a fake
git, so nothing is pulled, built or read from the network:

* rule 1, a pinned tag: an official one used as given (``edge`` pulled every call),
  found on this machine when the registry is unreachable, refused when missing, refused
  when the web app must be built; a ``local-*`` one built only where missing;
* rule 2, a clean checkout: ``edge`` when every image's revision label is HEAD, else
  ``sha-<7>`` (on GHCR, or on this machine offline);
* rule 3, otherwise ``local-<sha7>[-dirty]``: clean contexts re-tagged from ``tree-<12>``,
  dirty or unpublished ones built, each with its reason;
* the frontend's build args and the port misconfiguration, from ``.env`` and the shell;
* ``make build``'s tag, the ``.env`` line make owns, the git reads, the registry adapter,
  and what ``make up`` runs for a plan.
"""
from __future__ import annotations

import importlib.util
import io
import os
import stat
import sys
import urllib.error
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("stack_images", _REPO / "scripts" / "stack_images.py")
stack = importlib.util.module_from_spec(_SPEC)
sys.modules["stack_images"] = stack  # before exec: dataclasses look their module up
_SPEC.loader.exec_module(stack)

HEAD = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
SHA = "sha-0d9f0ac"
LOCAL = "local-0d9f0ac"
TREES = {"orchestrator": "1" * 40, "frontend": "2" * 40, "services/workspace-worker": "3" * 40}
NAMES = {"backend": "api", "frontend": "frontend", "workspace-worker": "workspace-worker"}
CONTEXTS = {"backend": "orchestrator", "frontend": "frontend", "workspace-worker": "services/workspace-worker"}
SERVICES = ("backend", "frontend", "workspace-worker")


def ref(service: str, tag: str) -> str:
    return f"ghcr.io/automatosai/automatos-{NAMES[service]}:{tag}"


def tree(service: str) -> str:
    return ref(service, "tree-" + TREES[CONTEXTS[service]][:12])


def every(tag: str) -> set:
    return {ref(service, tag) for service in SERVICES}


class FakeLookup:
    """This machine's images, the registry's tags (or unreachable ones) and edge's labels."""

    def __init__(self, local=(), published=(), unreachable=(), revisions=None):
        self._local, self._published = set(local), set(published)
        self._unreachable, self._revisions = set(unreachable), dict(revisions or {})

    def local(self, image: str) -> bool:
        return image in self._local

    def published(self, image: str):
        return None if image in self._unreachable else image in self._published

    def revision(self, image: str):
        return self._revisions.get(image)


def checkout(dirty=(), head=HEAD, trees=None) -> "stack.Checkout":
    return stack.Checkout(head, frozenset(dirty), dict(TREES if trees is None else trees))


CLEAN = checkout()
FRONTEND_REASON = ["NEXT_PUBLIC_API_URL=http://192.168.1.5:8000 (the published web app has http://localhost:8000)"]


# --------------------------------------------------------------------------- rule 1: a pinned tag


def test_a_pinned_official_tag_is_used_as_given_and_nothing_is_pulled_ahead():
    plan = stack.plan_up("sha-1234567", CLEAN, [], FakeLookup(published=every("sha-1234567")))
    assert (plan.tag, plan.pull, plan.retag, plan.build) == ("sha-1234567", False, (), ())
    assert "set in the shell or on the make command line" in plan.reason


def test_a_pinned_edge_is_pulled_on_every_call():
    plan = stack.plan_up("edge", CLEAN, [], FakeLookup(local=every("edge"), published=every("edge")))
    assert (plan.tag, plan.pull, plan.build) == ("edge", True, ())


def test_a_pinned_official_tag_ghcr_lacks_stops_make_up():
    lookup = FakeLookup(published=every("sha-1234567") - {ref("frontend", "sha-1234567")})
    with pytest.raises(stack.Stop, match=r"automatos-frontend:sha-1234567 \(ghcr.io does not have it\)"):
        stack.plan_up("sha-1234567", CLEAN, [], lookup)


def test_a_pinned_tag_on_this_machine_runs_offline():
    lookup = FakeLookup(local=every("sha-1234567"), unreachable=every("sha-1234567"))
    assert stack.plan_up("sha-1234567", CLEAN, [], lookup).tag == "sha-1234567"


def test_a_pinned_tag_the_registry_cannot_confirm_stops_make_up():
    with pytest.raises(stack.Stop, match="ghcr.io could not be reached"):
        stack.plan_up("sha-1234567", CLEAN, [], FakeLookup(unreachable=every("sha-1234567")))


def test_a_pinned_local_tag_builds_only_what_this_machine_lacks():
    some = FakeLookup(local={ref("backend", "local-feat-x")})
    plan = stack.plan_up("local-feat-x", CLEAN, [], some)
    assert plan.tag == "local-feat-x" and plan.pull is False and plan.retag == ()
    assert [service for service, _ in plan.build] == ["frontend", "workspace-worker"]
    assert stack.plan_up("local-feat-x", CLEAN, [], FakeLookup(local=every("local-feat-x"))).build == ()


def test_a_pinned_official_tag_cannot_carry_a_web_app_built_here():
    with pytest.raises(stack.Stop, match="never builds under an official tag such as sha-1234567"):
        stack.plan_up("sha-1234567", CLEAN, FRONTEND_REASON, FakeLookup(published=every("sha-1234567")))


# --------------------------------------------------------------------------- rule 2: a clean checkout


def test_a_clean_checkout_runs_edge_when_edge_is_a_build_of_head():
    revisions = {image: HEAD for image in every("edge")}
    plan = stack.plan_up("", CLEAN, [], FakeLookup(published=every(SHA), revisions=revisions))
    assert (plan.tag, plan.pull, plan.build) == ("edge", True, ())
    assert "revision label is HEAD" in plan.reason


def test_a_clean_checkout_runs_its_sha_when_any_edge_image_is_another_commit():
    revisions = {image: HEAD for image in every("edge")}
    revisions[ref("workspace-worker", "edge")] = "f" * 40
    plan = stack.plan_up("", CLEAN, [], FakeLookup(published=every(SHA), revisions=revisions))
    assert (plan.tag, plan.pull, plan.retag, plan.build) == (SHA, False, (), ())
    assert plan.reason == f"this checkout is clean and CI published {SHA} for its commit"


def test_a_sha_tag_on_this_machine_counts_as_found_offline():
    lookup = FakeLookup(local=every(SHA), unreachable=every(SHA) | every("edge"))
    assert stack.plan_up("", CLEAN, [], lookup).tag == SHA


# --------------------------------------------------------------------------- rule 3: a local tag


def test_an_unpublished_commit_re_tags_every_published_tree_and_builds_nothing():
    lookup = FakeLookup(published={tree(service) for service in SERVICES})
    plan = stack.plan_up("", CLEAN, [], lookup)
    assert plan.tag == LOCAL and plan.build == ()
    assert plan.retag == tuple((service, tree(service)) for service in SERVICES)
    assert "CI has not published sha-0d9f0ac" in plan.reason


def test_a_dirty_context_is_built_and_the_clean_ones_re_tagged():
    lookup = FakeLookup(published={tree(service) for service in SERVICES} | every(SHA))
    plan = stack.plan_up("", checkout(dirty={"orchestrator"}), [], lookup)
    assert plan.tag == f"{LOCAL}-dirty"
    assert plan.build == (("backend", "orchestrator has uncommitted changes"),)
    assert plan.retag == (("frontend", tree("frontend")), ("workspace-worker", tree("workspace-worker")))
    assert plan.reason == "orchestrator has uncommitted changes"


def test_a_tree_ci_never_published_is_built_and_says_so():
    lookup = FakeLookup(published={tree("backend"), tree("workspace-worker")})
    plan = stack.plan_up("", CLEAN, [], lookup)
    ((service, why),) = plan.build
    assert service == "frontend" and f"no tree-{'2' * 12}" in why


def test_a_tree_tag_on_this_machine_counts_as_found():
    lookup = FakeLookup(local={tree(service) for service in SERVICES})
    assert stack.plan_up("", CLEAN, [], lookup).build == ()


def test_a_tree_git_could_not_read_is_built():
    lookup = FakeLookup(published={tree(service) for service in SERVICES})
    plan = stack.plan_up("", checkout(trees={**TREES, "frontend": ""}), [], lookup)
    assert plan.build == (("frontend", "the tree of frontend could not be read from git"),)


def test_other_build_args_build_the_web_app_and_re_tag_the_rest_from_the_sha():
    plan = stack.plan_up("", CLEAN, FRONTEND_REASON, FakeLookup(published=every(SHA)))
    assert plan.tag == LOCAL
    assert plan.retag == (("backend", ref("backend", SHA)), ("workspace-worker", ref("workspace-worker", SHA)))
    ((service, why),) = plan.build
    assert service == "frontend" and FRONTEND_REASON[0] in why
    assert f"the rest is {SHA}, re-tagged" in plan.reason


def test_outside_a_git_checkout_make_up_runs_edge_and_says_why():
    plan = stack.plan_up("", checkout(head=None), [], FakeLookup(published=every("edge")))
    assert (plan.tag, plan.pull) == ("edge", True) and "not a git checkout" in plan.reason
    with pytest.raises(stack.Stop):
        stack.plan_up("", checkout(head=None), FRONTEND_REASON, FakeLookup(published=every("edge")))


# --------------------------------------------------------------------------- make build


@pytest.mark.parametrize("pinned,state,expected", [
    ("", CLEAN, LOCAL),
    ("", checkout(dirty={"frontend"}), f"{LOCAL}-dirty"),
    ("", checkout(head=None), "local-unversioned"),
    ("local-feat-x", CLEAN, "local-feat-x"),
])
def test_make_build_tags_its_builds_local(pinned, state, expected):
    assert stack.plan_build(pinned, state) == expected


def test_make_build_never_builds_under_an_official_tag():
    with pytest.raises(stack.Stop, match="never builds under an official tag"):
        stack.plan_build("edge", CLEAN)


# --------------------------------------------------------------------------- settings


def test_the_shipped_env_example_needs_no_web_app_build_and_no_fix():
    values = stack.settings(stack.read_dotenv((_REPO / ".env.example").read_text(encoding="utf-8")), {})
    assert stack.frontend_reasons(values) == []
    assert stack.misconfigurations(values) == []


def test_a_different_api_address_builds_the_web_app():
    values = stack.settings({"NEXT_PUBLIC_API_URL": "http://192.168.1.5:8000"}, {})
    assert stack.frontend_reasons(values) == FRONTEND_REASON


def test_the_shell_wins_over_env_as_compose_reads_them():
    values = stack.settings({"NEXT_PUBLIC_API_URL": "http://192.168.1.5:8000"}, {"NEXT_PUBLIC_API_URL": ""})
    assert stack.frontend_reasons(values) == []


@pytest.mark.parametrize("dotenv,expected", [
    ({"API_PORT": "8001"}, "API_PORT=8001, but the web app would still call port 8000: set "
                           "NEXT_PUBLIC_API_URL=http://localhost:8001 and NEXT_PUBLIC_WS_URL=ws://localhost:8001/ws"),
    ({"API_PORT": "8001", "NEXT_PUBLIC_API_URL": "http://localhost:8001"},
     "set NEXT_PUBLIC_WS_URL=ws://localhost:8001/ws in .env"),
    ({"MINIO_PORT": "9100"}, "MINIO_PORT=9100, but the web app would still call port 9000: set "
                             "NEXT_PUBLIC_MEDIA_ORIGINS=http://localhost:9100 in .env"),
])
def test_a_moved_port_without_its_next_public_values_is_a_misconfiguration(dotenv, expected):
    (problem,) = stack.misconfigurations(stack.settings(dotenv, {}))
    assert expected in problem


def test_a_moved_port_with_its_values_set_is_fine_and_builds_the_web_app():
    dotenv = {"API_PORT": "8001", "NEXT_PUBLIC_API_URL": "http://localhost:8001",
              "NEXT_PUBLIC_WS_URL": "ws://localhost:8001/ws", "MINIO_PORT": "9100",
              "NEXT_PUBLIC_MEDIA_ORIGINS": "http://localhost:9100"}
    values = stack.settings(dotenv, {})
    assert stack.misconfigurations(values) == []
    assert len(stack.frontend_reasons(values)) == 3


def test_read_dotenv_reads_values_as_compose_does():
    text = "# c\nA=1\nexport B=two\nC='x # y'\nD=\"q\"\nE=v # note\nbroken\nA=3\n"
    assert stack.read_dotenv(text) == {"A": "3", "B": "two", "C": "x # y", "D": "q", "E": "v"}


@pytest.mark.parametrize("environ,expected", [({}, ""), ({"AUTOMATOS_IMAGE_TAG": "  "}, ""),
                                              ({"AUTOMATOS_IMAGE_TAG": "sha-1234567"}, "sha-1234567")])
def test_an_empty_pin_is_no_pin(environ, expected):
    assert stack.pinned_tag(environ) == expected


def test_a_pin_that_is_not_a_tag_stops():
    with pytest.raises(stack.Stop, match="is not an image tag"):
        stack.pinned_tag({"AUTOMATOS_IMAGE_TAG": "edge; rm -rf /"})


# --------------------------------------------------------------------------- the .env line make owns


def test_the_tag_line_is_appended_and_every_other_line_kept():
    text = "POSTGRES_PASSWORD=x\n# mine\nAPI_PORT=8000\n\n"
    out = stack.with_tag(text, SHA)
    assert out == f"POSTGRES_PASSWORD=x\n# mine\nAPI_PORT=8000\n\n{stack.TAG_COMMENT}\nAUTOMATOS_IMAGE_TAG={SHA}\n"


def test_each_run_replaces_the_line_it_wrote_and_a_hand_written_one():
    once = stack.with_tag("A=1\nAUTOMATOS_IMAGE_TAG=edge\n", SHA)
    twice = stack.with_tag(once, LOCAL)
    assert twice == f"A=1\n\n{stack.TAG_COMMENT}\nAUTOMATOS_IMAGE_TAG={LOCAL}\n"
    assert stack.with_tag(twice, LOCAL) == twice
    assert stack.with_tag("", SHA) == f"{stack.TAG_COMMENT}\nAUTOMATOS_IMAGE_TAG={SHA}\n"


def test_write_tag_keeps_the_files_mode_and_creates_a_private_one(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    env.chmod(0o640)
    stack.write_tag(env, SHA)
    assert stat.S_IMODE(env.stat().st_mode) == 0o640
    assert env.read_text(encoding="utf-8").endswith(f"AUTOMATOS_IMAGE_TAG={SHA}\n")
    fresh = tmp_path / "new" / ".env"
    fresh.parent.mkdir()
    stack.write_tag(fresh, LOCAL)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600
    assert sorted(path.name for path in fresh.parent.iterdir()) == [".env"], "no temporary file is left"


# --------------------------------------------------------------------------- git


def fake_git(answers: dict):
    def run(root, *args):
        return answers.get(args)
    return run


def test_read_checkout_reads_head_the_dirty_contexts_and_the_trees():
    answers = {("rev-parse", "--verify", "HEAD"): HEAD,
               ("status", "--porcelain", "--", "orchestrator"): " M orchestrator/main.py",
               ("status", "--porcelain", "--", "frontend"): "",
               ("status", "--porcelain", "--", "services/workspace-worker"): None}
    answers.update({("rev-parse", f"HEAD:{context}"): value for context, value in TREES.items()})
    found = stack.read_checkout(_REPO, run_git=fake_git(answers))
    assert found.head == HEAD and found.trees == TREES
    assert found.dirty == {"orchestrator", "services/workspace-worker"}, "an unreadable status counts as dirty"


def test_outside_a_git_checkout_there_is_no_head():
    assert stack.read_checkout(_REPO, run_git=fake_git({})).head is None


def test_the_tree_tag_is_twelve_characters_of_the_context_tree():
    git = fake_git({("rev-parse", "HEAD:orchestrator"): TREES["orchestrator"]})
    assert stack.tree_tag(_REPO, "orchestrator", run_git=git) == "tree-111111111111"
    with pytest.raises(stack.Stop):
        stack.tree_tag(_REPO, "nowhere", run_git=git)


# --------------------------------------------------------------------------- the registry


class FakeClient:
    """image_registry's resolve and image, answering from a table; an exception is raised."""

    READ_ERRORS = (urllib.error.URLError, OSError, ValueError, LookupError)

    def __init__(self, answers):
        self.answers = answers

    def _answer(self, image):
        answer = self.answers[image]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def resolve(self, image):
        return self._answer(image)

    def image(self, image, arch):
        return {}, self._answer(image), None


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://ghcr.io/v2/x", code, "status", {}, io.BytesIO(b""))


def test_the_registry_says_published_not_published_or_unknown(capsys):
    registry = stack.Registry(FakeClient({"a": "sha256:" + "a" * 64, "b": _http(404), "c": _http(503),
                                          "d": urllib.error.URLError("offline")}))
    assert [registry.published(image) for image in "abcd"] == [True, False, None, None]
    assert capsys.readouterr().err.count("could not read") == 2, "an unread tag is reported, never hidden"


def test_the_registry_reads_the_revision_label_without_pulling():
    label = {"config": {"Labels": {stack.REVISION_LABEL: HEAD}}}
    registry = stack.Registry(FakeClient({"edge": label, "gone": _http(404), "bare": {"config": {}}}))
    assert [registry.revision(image) for image in ("edge", "gone", "bare")] == [HEAD, None, None]


# --------------------------------------------------------------------------- what make up runs


class FakeDocker:
    """Records the docker commands make up would run; ``fail`` names the ones that fail."""

    def __init__(self, local=(), fail=()):
        self._local, self._fail, self.calls = set(local), set(fail), []

    def local(self, image):
        return image in self._local

    def run(self, *args, env=None):
        self.calls.append((args, (env or {}).get(stack.TAG_KEY)))
        return 1 if args[:2] in self._fail else 0

    def check(self, *args, env=None):
        if self.run(*args, env=env) != 0:
            raise stack.Stop(f"docker {' '.join(args)} failed")


def test_edge_is_pulled_through_compose_with_the_tag():
    docker = FakeDocker()
    stack.execute(stack.Plan("edge", "why", pull=True), docker, {stack.TAG_KEY: "edge"})
    assert docker.calls == [(("compose", "pull", *SERVICES), "edge")]


def test_a_failed_edge_pull_runs_the_edge_this_machine_has_or_stops(capsys):
    plan = stack.Plan("edge", "why", pull=True)
    stack.execute(plan, FakeDocker(local=every("edge"), fail={("compose", "pull")}), {})
    assert "could not pull edge" in capsys.readouterr().err
    with pytest.raises(stack.Stop, match="does not have the edge images"):
        stack.execute(plan, FakeDocker(fail={("compose", "pull")}), {})


def test_re_tags_pull_only_what_is_missing_and_builds_go_through_compose():
    plan = stack.Plan(LOCAL, "why", retag=(("backend", tree("backend")), ("frontend", ref("frontend", "edge"))),
                      build=(("workspace-worker", "because"),))
    docker = FakeDocker(local={tree("backend"), ref("frontend", "edge")})
    stack.execute(plan, docker, {stack.TAG_KEY: LOCAL})
    assert [args for args, _ in docker.calls] == [
        ("tag", tree("backend"), ref("backend", LOCAL)),
        ("pull", ref("frontend", "edge")),  # edge moves, so it is pulled even when present
        ("tag", ref("frontend", "edge"), ref("frontend", LOCAL)),
        ("compose", "build", "workspace-worker"),
    ]
    assert docker.calls[-1][1] == LOCAL, "compose builds under the plan's tag"


def test_make_up_says_the_tag_why_and_what_it_builds():
    lines = stack.describe(stack.Plan(LOCAL, "reason", build=(("frontend", "args differ"),)))
    assert lines[0] == f"make up: AUTOMATOS_IMAGE_TAG={LOCAL} (reason)"
    assert "  frontend: built here, because args differ" in lines
    assert "  nothing is built" in stack.describe(stack.Plan(SHA, "reason"))


def test_make_up_end_to_end_writes_the_tag_it_ran(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text("POSTGRES_PASSWORD=x\n", encoding="utf-8")
    docker = FakeDocker()
    monkeypatch.setattr(stack, "Docker", lambda: docker)
    monkeypatch.setattr(stack, "Registry", lambda: FakeLookup(published=every(SHA)))
    monkeypatch.setattr(stack, "read_checkout", lambda root: CLEAN)
    monkeypatch.setattr(stack.os, "environ", {k: v for k, v in os.environ.items() if k != stack.TAG_KEY})
    assert stack.main(["--root", str(tmp_path), "up"]) == 0
    assert (tmp_path / ".env").read_text(encoding="utf-8").endswith(f"AUTOMATOS_IMAGE_TAG={SHA}\n")
    assert docker.calls == [], "a published sha is started by compose up, with nothing pulled ahead or built"
    assert f"AUTOMATOS_IMAGE_TAG={SHA} (this checkout is clean" in capsys.readouterr().out


def test_a_stop_exits_2_with_the_reason(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text("API_PORT=8001\n", encoding="utf-8")
    monkeypatch.setattr(stack.os, "environ", {})
    assert stack.main(["--root", str(tmp_path), "up"]) == stack.STOP_EXIT
    assert "make up: API_PORT=8001, but the web app would still call port 8000" in capsys.readouterr().err
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "API_PORT=8001\n", ".env is untouched"
