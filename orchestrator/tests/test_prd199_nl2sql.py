"""PRD-199 — NL2SQL productization.

* S1 — the semantic-layer write path PERSISTS: live committed session
  (the old writer targeted the detached object ``_get_source`` returns),
  workspace-scoped (the old handler never passed the workspace — its own
  crash was the only thing stopping a cross-tenant write), and the two
  phantom validator methods are deleted, not stubbed.
* S2 — one canonical format: the READER's shape ({instructions,
  metrics: {name: {sql, description}}, dimensions: {category: {name:
  sql}}}), written by the service, served by the new GET route, authored
  by the instructions-first editor.

Pure — ORM/session mocked at the boundary; no live DB, no LLM.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))


# ---------------------------------------------------------------------------
# S1 — the write path persists, workspace-scoped, no phantom methods
# ---------------------------------------------------------------------------

def _service_cls():
    from modules.nl2sql.service import DatabaseKnowledgeService

    return DatabaseKnowledgeService


@pytest.mark.asyncio
async def test_update_semantic_layer_persists_and_reads_back(monkeypatch):
    """A save lands on a LIVE session (commit called), in the canonical
    reader shape, and the schema cache entry is invalidated."""
    import core.database.database as db_mod

    fake_source = SimpleNamespace(semantic_layer=None)
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.filter.return_value.first.return_value = fake_source
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: fake_db)

    svc = SimpleNamespace(schema_cache={"7": {"stale": True}})
    doc = await _service_cls().update_semantic_layer(
        svc,
        "7",
        {
            "instructions": "  active = status <> 'churned'  ",
            "metrics": {"revenue": {"sql": "SUM(orders.total)", "description": "gross"}},
            "dimensions": {"time": {"month": "date_trunc('month', created_at)"}},
        },
        workspace_id="ws-a",
    )

    assert fake_source.semantic_layer == doc
    assert doc["instructions"] == "active = status <> 'churned'"
    assert doc["metrics"]["revenue"]["sql"] == "SUM(orders.total)"
    assert doc["dimensions"]["time"]["month"] == "date_trunc('month', created_at)"
    assert "updated_at" in doc
    fake_db.commit.assert_called_once()
    fake_db.close.assert_called_once()
    assert "7" not in svc.schema_cache  # cache invalidated


@pytest.mark.asyncio
async def test_update_semantic_layer_is_workspace_scoped(monkeypatch):
    """A source outside the caller's workspace is unreachable — the
    workspace filter applies and a miss raises, never writes."""
    import core.database.database as db_mod

    fake_db = MagicMock()
    # Workspace-filtered query finds nothing (cross-tenant attempt).
    fake_db.query.return_value.filter.return_value.filter.return_value.first.return_value = None
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: fake_db)

    svc = SimpleNamespace(schema_cache={})
    with pytest.raises(ValueError, match="not found"):
        await _service_cls().update_semantic_layer(
            svc, "7", {"instructions": "x"}, workspace_id="ws-intruder"
        )
    fake_db.commit.assert_not_called()


def test_no_phantom_semantic_methods():
    """The two never-defined methods the old writer crashed on are gone —
    deleted, not reimplemented as stubs."""
    src = (_orchestrator_root / "modules" / "nl2sql" / "service.py").read_text()
    assert "_get_schema_context" not in src
    assert "_validate_semantic_definitions" not in src


@pytest.mark.asyncio
async def test_get_semantic_layer_returns_stored_or_empty_doc():
    """The GET path returns the stored doc, or the canonical empty shape —
    never None (the editor renders it directly)."""
    cls = _service_cls()

    stored = {"instructions": "x", "metrics": {}, "dimensions": {}}
    svc = SimpleNamespace(
        _get_source=AsyncMock(return_value=SimpleNamespace(semantic_layer=stored))
    )
    assert await cls.get_semantic_layer(svc, "7", workspace_id="ws-a") == stored

    svc_empty = SimpleNamespace(
        _get_source=AsyncMock(return_value=SimpleNamespace(semantic_layer=None))
    )
    empty = await cls.get_semantic_layer(svc_empty, "7", workspace_id="ws-a")
    assert empty == {"instructions": "", "metrics": {}, "dimensions": {}}
    # the load is workspace-scoped
    svc_empty._get_source.assert_awaited_once_with("7", workspace_id="ws-a")


# ---------------------------------------------------------------------------
# S2 — one canonical format = the reader's contract
# ---------------------------------------------------------------------------

def test_semantic_format_matches_reader_contract():
    """A doc built the way the API builds it is consumable by the reader's
    exact iteration (nl2sql_service.py) — dict .items(), metric['sql'],
    nested dimensions — with no AttributeError. This is the format mismatch
    that made the old writer's output unreadable (list + sql_expression)."""
    from api.database_knowledge import SemanticDimensionRow, SemanticMetricRow, _dimension_rows_to_doc

    metrics_rows = [SemanticMetricRow(name="revenue", sql="SUM(total)", description="gross")]
    dim_rows = [
        SemanticDimensionRow(category="time", name="month", sql="date_trunc('month', d)"),
        SemanticDimensionRow(category="time", name="year", sql="date_trunc('year', d)"),
        SemanticDimensionRow(category="geo", name="country", sql="customers.country"),
    ]
    doc = {
        "instructions": "active = status <> 'churned'",
        "metrics": {m.name: {"sql": m.sql, "description": m.description or ""} for m in metrics_rows},
        "dimensions": _dimension_rows_to_doc(dim_rows),
    }

    # Mirror the reader's exact consumption (nl2sql_service.py:199-218).
    rendered = []
    instructions = doc.get("instructions") or doc.get("business_context")
    assert instructions
    for name, metric in doc["metrics"].items():
        rendered.append(f"{name}: {metric.get('sql')} -- {metric.get('description', '')}")
    for category, dims in doc.get("dimensions", {}).items():
        for name, sql in dims.items():
            rendered.append(f"{category}.{name}: {sql}")

    assert rendered == [
        "revenue: SUM(total) -- gross",
        "time.month: date_trunc('month', d)",
        "time.year: date_trunc('year', d)",
        "geo.country: customers.country",
    ]


def test_semantic_body_defaults_are_empty_not_required():
    """The editor can save an instructions-only doc — rows are optional."""
    from api.database_knowledge import SemanticLayerBody

    body = SemanticLayerBody(instructions="only text")
    assert body.metrics == []
    assert body.dimensions == []


# ---------------------------------------------------------------------------
# S4 — the A/B harness computes ΔS = acc_with − acc_without (pure, no LLM)
# ---------------------------------------------------------------------------

def test_ab_eval_reports_delta():
    """Given scripted generators (semantic-on answers two goldens, off
    answers one), the A/B reports the arithmetic — treatment minus
    baseline, the honest-gate shape."""
    from tests.nl2sql_eval import harness

    questions = harness.load_questions()
    goldens = {q["question"]: (q["id"], q["golden_sql"]) for q in questions}

    def factory(semantic_doc):
        answered = {"q01", "q03"} if semantic_doc else {"q01"}

        def generate_fn(question, schema):
            qid, golden = goldens[question]
            return golden if qid in answered else "SELECT 1"

        return generate_fn

    ab = harness.evaluate_ab(factory)

    assert ab["total"] == len(questions)
    assert ab["with_correct"] == 2
    assert ab["without_correct"] == 1
    assert ab["delta_s"] == pytest.approx(1 / len(questions), abs=1e-3)


def test_semantic_fixture_is_reader_shaped():
    """The seeded A/B doc is consumable by the reader's exact iteration —
    the same contract the S1/S2 write path persists."""
    from tests.nl2sql_eval import harness

    doc = harness.load_semantic_fixture()
    assert str(doc.get("instructions") or "").strip()
    for name, metric in doc["metrics"].items():
        assert isinstance(name, str) and metric.get("sql")
    for category, dims in doc.get("dimensions", {}).items():
        for dim_name, sql in dims.items():
            assert isinstance(dim_name, str) and isinstance(sql, str)


# ---------------------------------------------------------------------------
# S5 — kill-list guards: the decoys stay dead (PRD-185-S5 source-grep shape)
# ---------------------------------------------------------------------------

import re  # noqa: E402

_NL2SQL = _orchestrator_root / "modules" / "nl2sql"

_KILLED_FILES = [
    "modules/nl2sql/intelligence/__init__.py",
    "modules/nl2sql/intelligence/agent.py",
    "modules/nl2sql/intelligence/clarifier.py",
    "modules/nl2sql/intelligence/explainer.py",
    "modules/nl2sql/intelligence/rephraser.py",
    "modules/nl2sql/intelligence/visualizer.py",
    "modules/nl2sql/query/schema_linker.py",
]

_KILLED_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+[\w.]*"
    r"(?:nl2sql\.intelligence\b|schema_linker\b"
    r"|SchemaLinker|SmartNL2SQLAgent|create_smart_agent)",
    re.M,
)


def test_no_intelligence_files_remain():
    for rel in _KILLED_FILES:
        assert not (_orchestrator_root / rel).exists(), f"{rel} resurrected"


def test_no_intelligence_or_schemalinker_imports_remain():
    """No import statement anywhere may reference the deleted decoys —
    comment mentions stay allowed; imports do not."""
    offenders = []
    for py in _orchestrator_root.rglob("*.py"):
        rel = py.relative_to(_orchestrator_root).as_posix()
        if rel.startswith((".venv", "venv", "node_modules")) or rel == "tests/test_prd199_nl2sql.py":
            continue
        if _KILLED_IMPORT.search(py.read_text(errors="ignore")):
            offenders.append(rel)
    assert offenders == []


def test_legacy_limb_removed():
    """The PRD-21 shadow pipeline (incl. the _build_connection_string crash
    path) is gone from service.py and stays gone."""
    src = (_NL2SQL / "service.py").read_text()
    for symbol in (
        "def _generate_sql",
        "def _validate_sql",
        "def _execute_query",
        "def _build_sql_generation_prompt",
        "def _extract_tables_from_sql",
        "def _create_agent_tools",
        "_build_connection_string",
    ):
        assert symbol not in src, f"legacy limb symbol back: {symbol}"


def test_example_store_training_stubs_removed():
    """add_ddl/add_documentation (Vanna-imitation, 0 callers, would pollute
    few-shot if ever called) are gone."""
    src = (_NL2SQL / "training" / "example_store.py").read_text()
    assert "def add_ddl" not in src
    assert "def add_documentation" not in src


def test_fake_stats_columns_removed():
    """The never-written stats columns are gone from the model and no prod
    code serializes them (real stats derive from database_query_audit)."""
    offenders = []
    for py in _orchestrator_root.rglob("*.py"):
        rel = py.relative_to(_orchestrator_root).as_posix()
        if rel.startswith((".venv", "venv", "node_modules", "alembic/", "tests/")):
            continue
        text = py.read_text(errors="ignore")
        if '"total_queries_executed"' in text or "total_queries_executed =" in text:
            offenders.append(rel)
    assert offenders == []
