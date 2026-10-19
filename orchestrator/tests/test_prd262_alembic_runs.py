"""PRD-262 US-029: one migration run per backend boot, held by the smoke.

The entrypoint is the only owner of migrations, and the image CMD runs no alembic.
The smokes (smoke-fresh-clone and smoke-images) prove it on a real stack:

* ``scripts/ci/smoke_alembic_behind.py`` stamps the database one revision behind head
  from inside the backend container, so one more boot has a real migration to run;
* ``scripts/ci/smoke_alembic.py`` reads the backend's log of every boot: the
  entrypoint's banner, no ``Context impl`` line after it, at most one alembic run (the
  PRD-262 §8 count), and exactly one on the behind-head boot;
* ``scripts/ci/smoke-fresh-clone.sh`` runs both after the warm restarts.

These tests hold the parts; the lanes run them only in CI.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_CI = _REPO / "scripts" / "ci"
_SMOKE = _CI / "smoke-fresh-clone.sh"
_WORKFLOWS = _REPO / ".github" / "workflows"
NS = 1_000_000_000
T0 = 1_792_100_000  # 2026-10-15 21:33:20 UTC, a whole epoch second
BACKEND, FRONTEND = "automatos_backend", "automatos_frontend"
CONTEXT_IMPL = "INFO  [alembic.runtime.migration] Context impl PostgresqlImpl."
BANNER = "🚀 Starting Backend Application"
STEP = "🗄️  Running database migrations (alembic upgrade heads)..."
FRESH_DONE = "init_fresh_db: done — 167 tables, alembic at heads."
STARTED = "INFO:     Application startup complete."


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _CI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


runs = _load("smoke_alembic")
behind = _load("smoke_alembic_behind")


def _at(offset_s: float) -> int:
    return T0 * NS + round(offset_s * 1000) * 1_000_000


def _started_at(ns: int) -> str:
    """Like ``State.StartedAt``: RFC 3339 with the fraction's trailing zeros trimmed."""
    seconds, fraction = divmod(ns, NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    digits = f"{fraction:09d}".rstrip("0")
    return f"{stamp}.{digits}Z" if digits else f"{stamp}Z"


def _log(at_s: float, text: str) -> str:
    """Like ``docker logs -t``: nine fractional digits, then the line."""
    seconds, fraction = divmod(_at(at_s), NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{fraction:09d}Z {text}"


def _boot_lines(*, fresh=False, migrations=1, banner=True, cmd_alembic=False) -> list[str]:
    """One boot's log: the entrypoint's steps, its banner, then the server."""
    lines = ["Automatos AI Backend Starting..."]
    if fresh:
        lines += [CONTEXT_IMPL] * 200 + [FRESH_DONE]  # init_fresh_db's replay: one per revision
    lines += [STEP, CONTEXT_IMPL] * migrations
    lines += [BANNER] if banner else []
    lines += [CONTEXT_IMPL] if cmd_alembic else []  # what the old CMD's alembic logged
    return lines + [STARTED]


def _record(label: str, container: str, start_s: float) -> dict:
    return {"label": label, "container": container, "started_at": _started_at(_at(start_s))}


def _problems(label: str, lines: list[str], expected=None) -> list[str]:
    return runs.boot_problems(label, runs.boot_facts(lines), expected)


# --------------------------------------------------------------------------- one boot


def test_a_boot_that_migrates_once_through_the_entrypoint_passes():
    facts = runs.boot_facts(_boot_lines())
    assert (facts["runs"], facts["migration_steps"], facts["context_impl_after_banner"]) == (1, 1, 0)
    assert facts["banner"] is True
    assert _problems("warm-1", _boot_lines()) == []


def test_the_fresh_replay_is_not_an_alembic_run():
    """init_fresh_db logs a Context impl line per revision before `init_fresh_db: done` (§8)."""
    facts = runs.boot_facts(_boot_lines(fresh=True))
    assert (facts["runs"], facts["context_impl_lines"], facts["init_fresh_db"]) == (1, 201, True)
    assert _problems("fresh", _boot_lines(fresh=True)) == []


def test_alembic_after_the_banner_means_the_cmd_ran_it():
    problems = _problems("warm-1", _boot_lines(cmd_alembic=True))
    assert any("after the banner, so the CMD ran alembic" in problem for problem in problems), problems


def test_a_boot_without_the_banner_ran_no_boot_step():
    """The stage's CMD was not taken for a server command, so nothing migrated before it."""
    problems = _problems("fresh", _boot_lines(migrations=0, banner=False))
    assert problems == ["fresh: no `Starting Backend Application` banner, so the entrypoint ran no boot step "
                        "before the server"]


def test_two_runs_in_one_boot_fail():
    problems = _problems("warm-2", _boot_lines(migrations=2))
    assert any("2 alembic run(s) and 2 `Running database migrations` line(s); at most 1" in p for p in problems)


@pytest.mark.parametrize("migrations,ok", [(0, False), (1, True)])
def test_the_expected_count_is_exact(migrations, ok):
    problems = _problems("behind-head", _boot_lines(migrations=migrations), expected=1)
    assert (problems == []) is ok, problems


# --------------------------------------------------------------------------- a run


RECORDS = [
    _record("fresh", BACKEND, 0.0),
    _record("frontend", FRONTEND, 50.0),
    _record("warm-1", BACKEND, 100.0),
    _record("behind-head", BACKEND, 200.0),
]


def _run_log(behind_migrations: int = 1) -> list[str]:
    boots = [(0.0, _boot_lines(fresh=True)), (100.0, _boot_lines()),
             (200.0, _boot_lines(migrations=behind_migrations))]
    return [_log(start + index * 0.01, text) for start, lines in boots for index, text in enumerate(lines)]


def test_the_log_is_cut_at_each_backend_boot_and_the_frontend_is_ignored():
    result = runs.check(RECORDS, _run_log(), {"behind-head": 1}, BACKEND)
    assert result["failures"] == []
    assert [(boot["label"], boot["runs"]) for boot in result["boots"]] == [("fresh", 1), ("warm-1", 1),
                                                                           ("behind-head", 1)]


def test_a_behind_head_boot_that_ran_no_migration_fails():
    result = runs.check(RECORDS, _run_log(behind_migrations=0), {"behind-head": 1}, BACKEND)
    assert result["failures"] == ["behind-head: expected 1 alembic run(s) and 1 `Running database migrations` "
                                  "line(s), found 0 and 0"]


def test_an_expected_boot_that_never_ran_fails():
    result = runs.check(RECORDS[:2], _run_log()[:10], {"behind-head": 1}, BACKEND)
    assert "behind-head: no such backend boot was timed" in result["failures"]
    assert runs.check([], [], {}, BACKEND)["failures"] == ["no backend boot was timed"]


def _write_run(tmp_path: Path, log: list[str]) -> list[str]:
    results, backend_log = tmp_path / "boots.jsonl", tmp_path / "backend.log"
    results.write_text("".join(json.dumps(record) + "\n" for record in RECORDS), encoding="utf-8")
    backend_log.write_text("\n".join(log) + "\n", encoding="utf-8")
    return ["--results", str(results), "--backend-log", str(backend_log), "--expect", "behind-head=1",
            "--out", str(tmp_path / "alembic-runs.json"), "--summary", str(tmp_path / "summary.md")]


@pytest.mark.parametrize("behind_migrations,code,verdict", [(1, 0, "passed"), (2, 1, "FAILED")])
def test_the_command_writes_its_record_and_summary(tmp_path, behind_migrations, code, verdict):
    assert runs.main(_write_run(tmp_path, _run_log(behind_migrations))) == code
    record = json.loads((tmp_path / "alembic-runs.json").read_text(encoding="utf-8"))
    assert [boot["label"] for boot in record["boots"]] == ["fresh", "warm-1", "behind-head"]
    assert record["expected"] == {"behind-head": 1}
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert f"### {runs.TITLE}: {verdict}" in summary
    assert "| behind-head |" in summary


@pytest.mark.parametrize("text", ["behind-head", "=1", "behind-head=one", "behind-head="])
def test_expect_takes_label_equals_count(text):
    with pytest.raises(SystemExit):
        runs.build_parser().parse_args(["--results", "r", "--backend-log", "l", "--expect", text])


# --------------------------------------------------------------------------- the stamp behind head


class _Script:
    def __init__(self, heads, down_revision):
        self._heads, self._down = tuple(heads), down_revision

    def get_heads(self):
        return self._heads

    def get_revision(self, revision):
        return types.SimpleNamespace(revision=revision, down_revision=self._down)


def _stamp(monkeypatch, heads, down_revision) -> tuple[int, list]:
    """Run the stamp against a fake script directory; (exit code, the stamps it made)."""
    stamps = []
    monkeypatch.setattr(behind, "Config", lambda path: f"config:{path}")
    monkeypatch.setattr(behind, "ScriptDirectory",
                        types.SimpleNamespace(from_config=lambda config: _Script(heads, down_revision)))
    monkeypatch.setattr(behind, "command", types.SimpleNamespace(
        stamp=lambda config, revisions, **kw: stamps.append((config, revisions, kw))))
    return behind.main(), stamps


@pytest.mark.parametrize("down_revision,parents", [("p", ["p"]), (("a", "b"), ["a", "b"])])
def test_the_head_is_stamped_back_to_its_parents_with_purge(monkeypatch, down_revision, parents):
    code, stamps = _stamp(monkeypatch, ["head"], down_revision)
    assert code == 0
    assert stamps == [("config:alembic.ini", parents, {"purge": True})]


@pytest.mark.parametrize("heads,down_revision", [(["h1", "h2"], "p"), (["base"], None)])
def test_nothing_is_stamped_without_one_head_and_a_parent(monkeypatch, heads, down_revision):
    code, stamps = _stamp(monkeypatch, heads, down_revision)
    assert code == 1 and stamps == []


# --------------------------------------------------------------------------- the wiring


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_smoke_boots_behind_head_and_checks_every_boot_after_the_warm_restarts():
    text = _SMOKE.read_text(encoding="utf-8")
    assert 'ALEMBIC_TOOL="scripts/ci/smoke_alembic.py"' in text
    assert 'BEHIND_TOOL="scripts/ci/smoke_alembic_behind.py"' in text
    stamp = _function(text, "boot_behind_head")
    order = [stamp.index(call) for call in ('compose exec -T backend python - < "$BEHIND_TOOL"',
                                            "compose restart backend", 'wait_backend "$BEHIND_HEAD_LABEL"')]
    assert order == sorted(order)
    check = _function(text, "check_alembic_runs")
    assert 'docker logs -t "$BACKEND_CONTAINER"' in check
    assert '--expect "${BEHIND_HEAD_LABEL}=1"' in check and "|| fail " in check
    main = _function(text, "main")
    order = [main.index(call) for call in ("warm_restarts", "boot_behind_head", "check_alembic_runs", 'outcome="pass"')]
    assert order == sorted(order)


@pytest.mark.parametrize("workflow,events", [("smoke-fresh-clone.yml", ("push", "pull_request")),
                                             ("smoke-images.yml", ("pull_request",))])
def test_both_smokes_watch_the_new_tools(workflow, events):
    data = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True
    for event in events:
        assert {"scripts/ci/smoke_alembic.py", "scripts/ci/smoke_alembic_behind.py"} <= set(triggers[event]["paths"])
