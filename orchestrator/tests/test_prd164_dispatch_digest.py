"""PRD-164 S4 (Q22) — field-digest dispatch replaces the 8K upstream stuffing.

Before S4 the coordinator stuffed every upstream dependency's raw output into
the next task's prompt at up to ``_MAX_UPSTREAM_CHARS = 8000`` chars EACH
("## Previous Task Outputs"). S4 deletes that channel: upstream knowledge now
reaches the dispatch prompt as the PRD-166 field digest — immediate dependency
outputs merged ahead of semantic field hits, the whole block budgeted by
``Config.FIELD_QUERY_TOKEN_BUDGET`` (the per-task budget), and the agent keeps
``platform_field_query`` for anything the budget dropped.

AC1: dispatch prompt size drops >=60% on a multi-task fixture while the
golden task still passes (the load-bearing upstream fact reaches the prompt).
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import re
import sys as _sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — the port points at nothing so the
# modules.tools import chain's fail-soft DB connect refuses instantly instead
# of hanging on a wedged local proxy. CI exports real POSTGRES_* so these
# setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from config import Config  # noqa: E402
from modules.context import field_scoring  # noqa: E402
from modules.context.field_scoring import (  # noqa: E402
    budget_results,
    estimate_tokens,
    format_digest,
    merge_dispatch_rows,
)
from modules.coordination.dispatcher import MissionDispatcher  # noqa: E402


ORCH_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixture — the multi-task mission: three verbose upstream outputs, each
# carrying one load-bearing fact, one polluted with a base64 image blob.
# ---------------------------------------------------------------------------

GOLDEN_FACT = "GOLDEN-FACT: use the v2 /payments API with cursor pagination"


def _verbose_output(marker: str, *, chars: int = 8200,
                    with_base64: bool = False) -> str:
    filler_unit = (
        "The agent examined the source material in detail and recorded "
        "intermediate observations about endpoints, rate limits and auth. "
    )
    head = f"{marker}\n\n"
    body = filler_unit * (chars // len(filler_unit) + 1)
    if with_base64:
        body = "data:image/png;base64," + ("A" * 2000) + "==\n" + body
    return (head + body)[:chars]


UPSTREAMS = [
    {"title": "Research payments API",
     "output": _verbose_output(GOLDEN_FACT)},
    {"title": "Audit legacy scraper",
     "output": _verbose_output("FACT-B: scraper breaks on JS-rendered pages",
                               with_base64=True)},
    {"title": "Collect auth requirements",
     "output": _verbose_output("FACT-C: OAuth client credentials flow")},
]


def _legacy_stuffed_prompt(task_title: str, description: str) -> str:
    """Replica of the pre-S4 dispatch prompt: build_task_prompt rendered every
    ``input_context['upstream_outputs']`` entry (already truncated to 8000
    chars each by _prepare_task) under '## Previous Task Outputs'."""
    parts = [f"# Task: {task_title}", f"\n{description}",
             "\n## Previous Task Outputs"]
    for u in UPSTREAMS:
        parts.append(f"\n### {u['title']}\n{u['output'][:8000]}")
    return "\n".join(parts)


def _digest_prompt(task_title: str, description: str) -> tuple[str, list]:
    """The S4 pipeline, end to end on pure seams: upstream rows (sanitized,
    value-capped like field injection) merged ahead of field hits, budgeted,
    formatted, then rendered by the REAL MissionDispatcher.build_task_prompt."""
    from services.coordinator_service import (
        FIELD_VALUE_CAP_CHARS,
        _sanitize_for_field,
    )

    upstream_rows = [
        {"key": u["title"],
         "value": _sanitize_for_field(u["output"])[:FIELD_VALUE_CAP_CHARS]}
        for u in UPSTREAMS
    ]
    # The field echoes one upstream output back (it was injected at task
    # completion under the same key) plus one cross-mission pattern.
    field_rows = [
        {"key": "Research payments API",
         "value": "stale echo of the injected output"},
        {"key": "workspace pattern",
         "value": "Prior mission: webhook retries need idempotency keys"},
    ]
    merged = merge_dispatch_rows(upstream_rows, field_rows)
    kept, truncated = budget_results(merged, Config.FIELD_QUERY_TOKEN_BUDGET)
    digest = format_digest(kept, truncated=truncated)

    task = SimpleNamespace(
        title=task_title,
        description=description,
        input_context={"field_digest": digest, "field_id": "field-1"},
        verification_criteria=None,
    )
    return MissionDispatcher.build_task_prompt(task), kept


# ---------------------------------------------------------------------------
# merge_dispatch_rows — pure merge contract
# ---------------------------------------------------------------------------


class TestMergeDispatchRows:
    def test_upstream_rows_lead_and_dedupe_field_echoes(self):
        upstream = [{"key": "Task A", "value": "fresh"}]
        field = [{"key": "task a", "value": "stale echo"},
                 {"key": "other", "value": "kept"}]
        merged = merge_dispatch_rows(upstream, field)
        assert merged[0] == {"key": "Task A", "value": "fresh"}
        assert {"key": "other", "value": "kept"} in merged
        assert all(r["value"] != "stale echo" for r in merged)

    def test_empty_inputs(self):
        assert merge_dispatch_rows([], []) == []
        only_field = [{"key": "k", "value": "v"}]
        assert merge_dispatch_rows([], only_field) == only_field
        only_up = [{"key": "k", "value": "v"}]
        assert merge_dispatch_rows(only_up, []) == only_up

    def test_does_not_mutate_inputs(self):
        upstream = [{"key": "a", "value": "1"}]
        field = [{"key": "b", "value": "2"}]
        merged = merge_dispatch_rows(upstream, field)
        merged.append({"key": "c", "value": "3"})
        assert upstream == [{"key": "a", "value": "1"}]
        assert field == [{"key": "b", "value": "2"}]


# ---------------------------------------------------------------------------
# AC1 — prompt size drops >=60%, golden fact still reaches the prompt
# ---------------------------------------------------------------------------


class TestDispatchPromptShrinks:
    TITLE = "Implement the payments integration"
    DESC = "Build the integration using everything learned upstream."

    def test_prompt_size_drops_at_least_60_percent(self):
        legacy = _legacy_stuffed_prompt(self.TITLE, self.DESC)
        new, _ = _digest_prompt(self.TITLE, self.DESC)
        assert len(legacy) > 24_000          # the fixture really is multi-task
        drop = 1 - (len(new) / len(legacy))
        assert drop >= 0.60, (
            f"dispatch prompt only dropped {drop:.0%} "
            f"({len(legacy)} -> {len(new)} chars)"
        )

    def test_golden_task_content_survives(self):
        """The golden check: the load-bearing upstream fact is still in the
        dispatch prompt, the task framing is intact, and the agent is told how
        to reach anything the budget dropped (field tools)."""
        new, _ = _digest_prompt(self.TITLE, self.DESC)
        assert GOLDEN_FACT in new
        assert self.TITLE in new
        assert self.DESC in new
        assert "platform_field_query" in new

    def test_digest_respects_per_task_token_budget(self):
        _, kept = _digest_prompt(self.TITLE, self.DESC)
        spent = sum(estimate_tokens(r["value"]) for r in kept)
        assert spent <= Config.FIELD_QUERY_TOKEN_BUDGET

    def test_stuffing_format_and_base64_are_gone(self):
        new, _ = _digest_prompt(self.TITLE, self.DESC)
        assert "## Previous Task Outputs" not in new
        assert "base64" not in new


# ---------------------------------------------------------------------------
# The stuffing is DELETED at source, not just bypassed (grep gates)
# ---------------------------------------------------------------------------


class TestStuffingDeletedAtSource:
    def _src(self, rel: str) -> str:
        return (ORCH_ROOT / rel).read_text(encoding="utf-8")

    def test_coordinator_no_longer_stuffs_upstream_outputs(self):
        src = self._src("services/coordinator_service.py")
        assert "_MAX_UPSTREAM_CHARS" not in src
        # the input_context dict-write of raw upstream outputs is gone
        assert '"upstream_outputs":' not in src
        # the digest pipeline is the replacement choke point
        assert "_collect_upstream_digest_rows" in src
        assert "merge_dispatch_rows" in src

    def test_dispatcher_prompt_no_longer_renders_stuffing(self):
        src = self._src("modules/coordination/dispatcher.py")
        assert "upstream_outputs" not in src
        assert "Previous Task Outputs" not in src
        # the digest render survives
        assert "field_digest" in src

    def test_matcher_keys_on_digest_not_stuffing(self):
        src = self._src("modules/coordination/agent_matcher.py")
        assert "upstream_outputs" not in src


# ---------------------------------------------------------------------------
# _collect_upstream_digest_rows — DB-shaped seam with a mocked session
# ---------------------------------------------------------------------------


def _mock_db(deps, dep_tasks):
    db = MagicMock()
    dep_q = MagicMock()
    dep_q.filter.return_value = dep_q
    dep_q.all.return_value = deps
    task_q = MagicMock()
    task_q.filter.return_value = task_q
    task_q.order_by.return_value = task_q
    task_q.all.return_value = dep_tasks
    calls = [0]

    def _route(model):
        calls[0] += 1
        return dep_q if calls[0] % 2 == 1 else task_q

    db.query.side_effect = _route
    return db


class TestCollectUpstreamDigestRows:
    def test_rows_are_capped_sanitized_and_keyed_by_title(self):
        from services.coordinator_service import (
            CoordinatorService,
            FIELD_VALUE_CAP_CHARS,
        )

        task = SimpleNamespace(id=uuid4())
        dep_task = SimpleNamespace(
            id=uuid4(), title="Upstream research", sequence_number=1,
            output=("data:image/png;base64," + "B" * 500 + "==\n"
                    + "finding " * 2000),
        )
        empty_task = SimpleNamespace(
            id=uuid4(), title="No output yet", sequence_number=2, output=None,
        )
        deps = [SimpleNamespace(task_id=task.id,
                                depends_on_task_id=dep_task.id),
                SimpleNamespace(task_id=task.id,
                                depends_on_task_id=empty_task.id)]

        rows = CoordinatorService._collect_upstream_digest_rows(
            _mock_db(deps, [dep_task, empty_task]), task)

        assert len(rows) == 1            # empty outputs are skipped
        assert rows[0]["key"] == "Upstream research"
        assert len(rows[0]["value"]) <= FIELD_VALUE_CAP_CHARS
        assert "base64" not in rows[0]["value"]
        assert "finding" in rows[0]["value"]

    def test_no_dependencies_returns_empty(self):
        from services.coordinator_service import CoordinatorService

        task = SimpleNamespace(id=uuid4())
        rows = CoordinatorService._collect_upstream_digest_rows(
            _mock_db([], []), task)
        assert rows == []


# ---------------------------------------------------------------------------
# _attach_field_digest — upstream rows flow in even with NO field backend
# ---------------------------------------------------------------------------


class TestAttachFieldDigestDegradedMode:
    @pytest.mark.asyncio
    async def test_upstream_rows_attach_without_field_backend(self):
        """Qdrant down/unconfigured: dependency context still reaches the
        prompt as a budgeted digest — never the raw 8K stuffing."""
        from services.coordinator_service import CoordinatorService

        svc = CoordinatorService.__new__(CoordinatorService)
        svc._field = None
        svc._get_field = lambda: None

        run = SimpleNamespace(id=uuid4(), config={}, tokens_used=0,
                              token_budget_estimate=10_000_000)
        task = SimpleNamespace(id=uuid4(), title="T", description="d",
                               input_context=None)
        rows = [{"key": "Upstream research", "value": GOLDEN_FACT}]

        await svc._attach_field_digest(MagicMock(), run, task, None, 1,
                                       upstream_rows=rows)

        digest = (task.input_context or {}).get("field_digest", "")
        assert GOLDEN_FACT in digest

    @pytest.mark.asyncio
    async def test_nothing_to_attach_writes_nothing(self):
        from services.coordinator_service import CoordinatorService

        svc = CoordinatorService.__new__(CoordinatorService)
        svc._field = None
        svc._get_field = lambda: None

        run = SimpleNamespace(id=uuid4(), config={}, tokens_used=0,
                              token_budget_estimate=10_000_000)
        task = SimpleNamespace(id=uuid4(), title="T", description="d",
                               input_context=None)

        await svc._attach_field_digest(MagicMock(), run, task, None, 1,
                                       upstream_rows=[])

        assert not (task.input_context or {}).get("field_digest")
